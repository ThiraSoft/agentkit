// ollama.go: the Ollama provider.

package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// ollamaMessage is the wire format expected by /api/chat. Ollama routes
// ALL multimodal blobs (image AND audio) through the same `images` field
// in base64 - there is no dedicated `audio` field on the API side (see
// issue ollama/ollama#15333 for Gemma 4 E4B audio).
type ollamaMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Images    []string   `json:"images,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	ToolID    string     `json:"tool_call_id,omitempty"`
	Name      string     `json:"name,omitempty"`
}

func toOllamaMessages(messages []Message) []ollamaMessage {
	out := make([]ollamaMessage, len(messages))
	for i, m := range messages {
		om := ollamaMessage{
			Role:      m.Role,
			Content:   m.Content,
			ToolCalls: m.ToolCalls,
			ToolID:    m.ToolID,
			Name:      m.Name,
		}
		for _, md := range m.Media {
			if md.Type == MediaImage || md.Type == MediaAudio {
				om.Images = append(om.Images, base64.StdEncoding.EncodeToString(md.Data))
			}
		}
		out[i] = om
	}
	return out
}

// ListOllamaModels queries the local Ollama server for available models.
// Returns nil if the server is unreachable - Ollama is optional.
func ListOllamaModels() []string {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get("http://localhost:11434/api/tags")
	if err != nil {
		return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil
	}

	var result struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil
	}

	models := make([]string, 0, len(result.Models))
	for _, m := range result.Models {
		models = append(models, m.Name)
	}
	return models
}

type ollamaProvider struct {
	baseURL    string
	Model      string
	client     *http.Client
	numCtx     int
	numPredict int
}

func newOllamaProvider(model string, numCtx, numPredict int) *ollamaProvider {
	return &ollamaProvider{
		baseURL:    "http://localhost:11434",
		Model:      model,
		client:     &http.Client{Timeout: 300 * time.Second},
		numCtx:     numCtx,
		numPredict: numPredict,
	}
}

// options builds the `options` field of /api/chat from the inference
// settings. Returns nil if there is nothing to set (Ollama then applies
// its own defaults). num_predict at 0 is deliberately omitted (0 = let
// Ollama decide; sending 0 would mean "generate no tokens").
func (p *ollamaProvider) options() map[string]any {
	opts := map[string]any{}
	if p.numCtx > 0 {
		opts["num_ctx"] = p.numCtx
	}
	if p.numPredict != 0 {
		opts["num_predict"] = p.numPredict
	}
	if len(opts) == 0 {
		return nil
	}
	return opts
}

func (p *ollamaProvider) ModelName() string { return p.Model }

func (p *ollamaProvider) Name() string { return "ollama" }

func (p *ollamaProvider) Chat(ctx context.Context, messages []Message, tools []Tool) (*Message, error) {
	reqBody := map[string]any{
		"model":    p.Model,
		"messages": toOllamaMessages(messages),
		"stream":   false,
	}
	if opts := p.options(); opts != nil {
		reqBody["options"] = opts
	}
	if len(tools) > 0 {
		reqBody["tools"] = tools
	}

	body, _ := json.Marshal(reqBody)
	req, _ := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/api/chat", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama error %d: %s", resp.StatusCode, body)
	}

	var result struct {
		Message Message `json:"message"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result.Message, nil
}

func (p *ollamaProvider) Stream(ctx context.Context, messages []Message, tools []Tool, onChunk func(string) error) (*Message, error) {
	reqBody := map[string]any{
		"model":    p.Model,
		"messages": toOllamaMessages(messages),
		"stream":   true,
	}
	if opts := p.options(); opts != nil {
		reqBody["options"] = opts
	}
	if len(tools) > 0 {
		reqBody["tools"] = tools
	}

	body, _ := json.Marshal(reqBody)
	req, _ := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/api/chat", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama stream error %d: %s", resp.StatusCode, body)
	}

	reader := bufio.NewReader(resp.Body)
	fullMessage := &Message{Role: "assistant"}

	for {
		// Proactive check before reading
		select {
		case <-ctx.Done():
			// Context is dead, return right away
			return fullMessage, ctx.Err()
		default:
			// Keep going
		}

		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return fullMessage, err
		}

		// Ollama sends direct JSON objects
		var chunk struct {
			Message struct {
				Role      string     `json:"role"`
				Content   string     `json:"content"`
				ToolCalls []ToolCall `json:"tool_calls"`
			} `json:"message"`
			Done bool `json:"done"`
		}

		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			continue
		}

		if chunk.Message.Content != "" {
			fullMessage.Content += chunk.Message.Content
			if onChunk != nil {
				if err := onChunk(chunk.Message.Content); err != nil {
					return fullMessage, err
				}
			}
		}

		if len(chunk.Message.ToolCalls) > 0 {
			fullMessage.ToolCalls = append(fullMessage.ToolCalls, chunk.Message.ToolCalls...)
		}

		if chunk.Done {
			break
		}
	}

	return fullMessage, nil
}
