// anthropic.go: the Anthropic provider.

package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

type anthropicProvider struct {
	apiKey  string
	baseURL string
	Model   string
	version string
	client  *http.Client
}

func newAnthropicProvider(model string) *anthropicProvider {
	return &anthropicProvider{
		apiKey:  os.Getenv("ANTHROPIC_API_KEY"),
		baseURL: "https://api.anthropic.com/v1",
		Model:   model,
		version: "2023-06-01",
		client:  &http.Client{Timeout: 300 * time.Second},
	}
}

func (p *anthropicProvider) ModelName() string { return p.Model }

func (p *anthropicProvider) Name() string { return "anthropic" }

func (p *anthropicProvider) Chat(ctx context.Context, messages []Message, tools []Tool) (*Message, error) {
	// Normalize for Anthropic
	system, msgs := p.normalizeMessages(messages)

	reqBody := map[string]any{
		"model":      p.Model,
		"max_tokens": 32000,
		"messages":   msgs,
	}

	if system != "" {
		reqBody["system"] = system
	}

	if len(tools) > 0 {
		reqBody["tools"] = p.convertTools(tools)
	}

	body, _ := json.Marshal(reqBody)
	req, _ := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", p.version)

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("anthropic error %d: %s", resp.StatusCode, body)
	}

	var result struct {
		Content []struct {
			Type  string         `json:"type"`
			Text  string         `json:"text,omitempty"`
			ID    string         `json:"id,omitempty"`
			Name  string         `json:"name,omitempty"`
			Input map[string]any `json:"input,omitempty"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return p.convertResponse(result), nil
}

func (p *anthropicProvider) Stream(ctx context.Context, messages []Message, tools []Tool, onChunk func(string) error) (*Message, error) {
	system, msgs := p.normalizeMessages(messages)

	reqBody := map[string]any{
		"model":      p.Model,
		"max_tokens": 32000,
		"messages":   msgs,
		"stream":     true,
	}

	if system != "" {
		reqBody["system"] = system
	}

	if len(tools) > 0 {
		reqBody["tools"] = p.convertTools(tools)
	}

	body, _ := json.Marshal(reqBody)
	req, _ := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", p.version)

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("anthropic stream error %d: %s", resp.StatusCode, body)
	}

	reader := bufio.NewReader(resp.Body)
	fullMessage := &Message{Role: "assistant"}

	// State variables for event parsing
	var currentToolID string
	var currentToolName string
	var currentToolInputBuffer string

	for {
		// Proactive check before reading
		select {
		case <-ctx.Done():
			// Context canceled, exit immediately
			return fullMessage, ctx.Err()
		default:
			// Continue
		}

		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return fullMessage, err
		}

		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		// Anthropic sometimes sends ping events or [DONE]

		var event struct {
			Type  string `json:"type"`
			Index int    `json:"index"`
			Delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJson string `json:"partial_json"`
			} `json:"delta"`
			ContentBlock struct {
				Type string `json:"type"`
				ID   string `json:"id"`
				Name string `json:"name"`
				Text string `json:"text"`
			} `json:"content_block"`
		}

		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "content_block_start":
			if event.ContentBlock.Type == "tool_use" {
				currentToolID = event.ContentBlock.ID
				currentToolName = event.ContentBlock.Name
				currentToolInputBuffer = ""
			} else if event.ContentBlock.Type == "text" {
				// Start of a text block
				if event.ContentBlock.Text != "" {
					fullMessage.Content += event.ContentBlock.Text
					if onChunk != nil {
						if err := onChunk(event.ContentBlock.Text); err != nil {
							return fullMessage, err
						}
					}
				}
			}

		case "content_block_delta":
			if event.Delta.Type == "text_delta" {
				fullMessage.Content += event.Delta.Text
				if onChunk != nil {
					if err := onChunk(event.Delta.Text); err != nil {
						return fullMessage, err
					}
				}
			} else if event.Delta.Type == "input_json_delta" {
				currentToolInputBuffer += event.Delta.PartialJson
			}

		case "content_block_stop":
			if currentToolID != "" {
				// End of a tool call
				fullMessage.ToolCalls = append(fullMessage.ToolCalls, ToolCall{
					ID:   currentToolID,
					Type: "function",
					Function: FunctionCall{
						Name:      currentToolName,
						Arguments: json.RawMessage(currentToolInputBuffer),
					},
				})
				currentToolID = ""
				currentToolName = ""
				currentToolInputBuffer = ""
			}

		case "message_stop":
			// End of message
		}
	}

	return fullMessage, nil
}

// normalizeMessages groups tool results into a single user message
func (p *anthropicProvider) normalizeMessages(messages []Message) (string, []map[string]any) {
	var system string
	var result []map[string]any

	for i := 0; i < len(messages); i++ {
		msg := messages[i]

		// Extract system prompt
		if msg.Role == "system" {
			system += msg.Content + "\n"
			continue
		}

		// Normal user/assistant message
		if msg.Role != "tool" {
			content := p.buildContent(msg)
			result = append(result, map[string]any{
				"role":    msg.Role,
				"content": content,
			})
			continue
		}

		// Tool results: group all subsequent ones
		toolResults := []any{}
		for i < len(messages) && messages[i].Role == "tool" {
			toolResults = append(toolResults, map[string]any{
				"type":        "tool_result",
				"tool_use_id": messages[i].ToolID,
				"content":     messages[i].Content,
			})
			i++
		}
		i-- // Compensate for main loop i++

		result = append(result, map[string]any{
			"role":    "user",
			"content": toolResults,
		})
	}

	return system, result
}

func (p *anthropicProvider) buildContent(msg Message) []any {
	content := []any{}

	// Text
	if msg.Content != "" {
		content = append(content, map[string]string{
			"type": "text",
			"text": msg.Content,
		})
	}

	// Tool uses
	for _, tc := range msg.ToolCalls {
		var input map[string]any
		json.Unmarshal(tc.Function.Arguments, &input)
		content = append(content, map[string]any{
			"type":  "tool_use",
			"id":    tc.ID,
			"name":  tc.Function.Name,
			"input": input,
		})
	}

	return content
}

func (p *anthropicProvider) convertTools(tools []Tool) []map[string]any {
	var result []map[string]any
	for _, t := range tools {
		result = append(result, map[string]any{
			"name":         t.Function.Name,
			"description":  t.Function.Description,
			"input_schema": t.Function.JSONSchema(),
		})
	}
	return result
}

func (p *anthropicProvider) convertResponse(result struct {
	Content []struct {
		Type  string         `json:"type"`
		Text  string         `json:"text,omitempty"`
		ID    string         `json:"id,omitempty"`
		Name  string         `json:"name,omitempty"`
		Input map[string]any `json:"input,omitempty"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
},
) *Message {
	msg := &Message{Role: "assistant"}

	for _, c := range result.Content {
		switch c.Type {
		case "text":
			msg.Content += c.Text
		case "tool_use":
			args, _ := json.Marshal(c.Input)
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:   c.ID,
				Type: "function",
				Function: FunctionCall{
					Name:      c.Name,
					Arguments: args,
				},
			})
		}
	}

	return msg
}
