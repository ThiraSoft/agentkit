// openai_compat.go: a base for OpenAI-compatible providers.
// OpenAI, Mistral, and VLLM all use the same chat/completions API format.

package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"slices"
	"strings"
	"time"
)

// openaiCompatProvider is the common base for OpenAI-compatible providers.
type openaiCompatProvider struct {
	apiKey       string
	baseURL      string
	model        string
	name         string
	client       *http.Client
	extraBody    map[string]any    // Additional fields in the body (tool_choice, etc.)
	extraHeaders map[string]string // Extra HTTP headers applied to every request
	audioFormat  string            // "audio_url" (vLLM/Gemma) or "input_audio" (OpenAI/llama.cpp)
	streamUsage  bool
	onReasoning  func(string)
	bodyFunc     func() map[string]any
}

// OpenAICompatConfig describes an OpenAI-compatible chat/completions
// endpoint for NewOpenAICompat.
type OpenAICompatConfig struct {
	APIKey  string // sent as "Authorization: Bearer <APIKey>" when not empty
	BaseURL string // up to and including the version, e.g. http://localhost:8000/v1
	Model   string
	Name    string        // returned by Name and used in error messages
	Timeout time.Duration // whole-request timeout, 600s when zero
	// ExtraBody adds raw fields to every request body (tool_choice,
	// temperature, reasoning_effort...). tool_choice is left out of
	// requests that carry no tools.
	ExtraBody map[string]any
	// ExtraBodyFunc, when set, is called for every request and its fields
	// are added after ExtraBody's, for what changes between two requests.
	ExtraBodyFunc func() map[string]any
	ExtraHeaders  map[string]string // extra HTTP headers on every request
	// AudioFormat selects the audio media encoding: "input_audio"
	// (OpenAI convention, expected by llama.cpp) or "audio_url" (vLLM/Gemma
	// recipes). Empty defaults to "audio_url".
	AudioFormat string
	// StreamUsage asks for the token usage at the end of a stream
	// (stream_options.include_usage). Some servers refuse the field.
	StreamUsage bool
	// OnReasoning, when set, receives what the model thinks before it
	// answers, as the server streams it apart from the answer
	// (reasoning_content, or reasoning). It is never part of the message.
	OnReasoning func(chunk string)
}

// NewOpenAICompat returns a provider for an OpenAI-compatible endpoint. It
// does not retry: wrap it with WithRetry to get the behaviour of NewProvider.
func NewOpenAICompat(cfg OpenAICompatConfig) Provider {
	return newOpenAICompatProvider(cfg)
}

func newOpenAICompatProvider(cfg OpenAICompatConfig) *openaiCompatProvider {
	if cfg.Timeout == 0 {
		cfg.Timeout = 600 * time.Second
	}
	if cfg.AudioFormat == "" {
		cfg.AudioFormat = "audio_url"
	}
	return &openaiCompatProvider{
		apiKey:       cfg.APIKey,
		baseURL:      cfg.BaseURL,
		model:        cfg.Model,
		name:         cfg.Name,
		client:       &http.Client{Timeout: cfg.Timeout},
		extraBody:    cfg.ExtraBody,
		extraHeaders: cfg.ExtraHeaders,
		audioFormat:  cfg.AudioFormat,
		streamUsage:  cfg.StreamUsage,
		onReasoning:  cfg.OnReasoning,
		bodyFunc:     cfg.ExtraBodyFunc,
	}
}

// openaiUsage is the usage object of chat/completions.
type openaiUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func (u *openaiUsage) usage() *Usage {
	if u == nil {
		return nil
	}
	return &Usage{
		InputTokens:     u.PromptTokens,
		OutputTokens:    u.CompletionTokens,
		CacheReadTokens: u.PromptTokensDetails.CachedTokens,
	}
}

func (p *openaiCompatProvider) applyExtraHeaders(req *http.Request) {
	for k, v := range p.extraHeaders {
		req.Header.Set(k, v)
	}
}

func (p *openaiCompatProvider) ModelName() string { return p.model }

func (p *openaiCompatProvider) Name() string { return p.name }

// openaiToolCall is the wire format for OpenAI-compatible APIs where
// arguments must be a JSON string, not an inline object.
type openaiToolCall struct {
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type"`
	Function openaiToolFunction `json:"function"`
}

type openaiToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // Must be a JSON string, not an object
}

// openaiMessage is the wire format sent to OpenAI-compatible APIs.
// It mirrors Message but uses openaiToolCall to ensure arguments is a string.
// Content is `any` so it can be either a plain string or an array of content
// blocks (text / image_url / input_audio) for multimodal requests.
type openaiMessage struct {
	Role      string           `json:"role"`
	Content   any              `json:"content"`
	ToolCalls []openaiToolCall `json:"tool_calls,omitempty"`
	ToolID    string           `json:"tool_call_id,omitempty"`
	Name      string           `json:"name,omitempty"`
}

// mediaToContentBlocks converts internal Media items to the OpenAI-compatible
// content block array. Audio → `audio_url` (data URL), images → `image_url`
// (data URL), same pattern, aligned with the vLLM Gemma 4 recipes. Text
// is placed AFTER media (Gemma 4 multimodal best practice).
func mediaToContentBlocks(text string, media []Media, audioFormat string) []map[string]any {
	blocks := make([]map[string]any, 0, len(media)+1)
	for _, m := range media {
		b64 := base64.StdEncoding.EncodeToString(m.Data)
		switch m.Type {
		case MediaAudio:
			mime := m.MimeType
			if mime == "" {
				mime = "audio/wav"
			}
			if audioFormat == "input_audio" {
				// Norme OpenAI / llama.cpp : bloc input_audio {data, format}.
				blocks = append(blocks, map[string]any{
					"type": "input_audio",
					"input_audio": map[string]any{
						"data":   b64,
						"format": audioMimeToFormat(mime),
					},
				})
				break
			}
			blocks = append(blocks, map[string]any{
				"type": "audio_url",
				"audio_url": map[string]any{
					"url": "data:" + mime + ";base64," + b64,
				},
			})
		case MediaImage:
			mime := m.MimeType
			if mime == "" {
				mime = "image/jpeg"
			}
			blocks = append(blocks, map[string]any{
				"type": "image_url",
				"image_url": map[string]any{
					"url": "data:" + mime + ";base64," + b64,
				},
			})
		}
	}
	if text != "" {
		blocks = append(blocks, map[string]any{"type": "text", "text": text})
	}
	return blocks
}

// audioMimeToFormat translates an audio MIME type into the "format" value
// expected by the input_audio block of the OpenAI/llama.cpp API.
func audioMimeToFormat(mime string) string {
	switch mime {
	case "audio/mpeg", "audio/mp3":
		return "mp3"
	case "audio/wav", "audio/x-wav", "audio/wave":
		return "wav"
	case "audio/flac":
		return "flac"
	case "audio/ogg":
		return "ogg"
	default:
		return "wav"
	}
}

// toOpenAIMessages converts internal Messages to the wire format expected by
// OpenAI-compatible APIs, ensuring tool_calls[].function.arguments is always
// a JSON string and type is always "function".
// It also sanitizes the history to remove incomplete tool sequences.
func toOpenAIMessages(messages []Message, audioFormat string) []openaiMessage {
	clean := sanitizeMessages(messages)
	out := make([]openaiMessage, len(clean))
	for i, m := range clean {
		var content any = m.Content
		if len(m.Media) > 0 {
			content = mediaToContentBlocks(m.Content, m.Media, audioFormat)
		}
		out[i] = openaiMessage{
			Role:    m.Role,
			Content: content,
			ToolID:  m.ToolID,
			Name:    m.Name,
		}
		if len(m.ToolCalls) > 0 {
			out[i].ToolCalls = make([]openaiToolCall, len(m.ToolCalls))
			for j, tc := range m.ToolCalls {
				args := string(tc.Function.Arguments)
				if args == "" {
					args = "{}"
				}
				out[i].ToolCalls[j] = openaiToolCall{
					ID:   tc.ID,
					Type: "function",
					Function: openaiToolFunction{
						Name:      tc.Function.Name,
						Arguments: args,
					},
				}
			}
		}
	}
	return out
}

// sanitizeMessages cleans up the message history to ensure it's valid for
// OpenAI-compatible APIs. It handles:
//  1. Truncated tool sequences (assistant has tool_calls but responses are missing)
//  2. Orphan tool messages (tool response without matching assistant tool_call)
//  3. Missing assistant after tool (some providers like vLLM require assistant
//     between tool responses and the next user message)
func sanitizeMessages(messages []Message) []Message {
	out := make([]Message, 0, len(messages))

	for i, m := range messages {
		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			// Collect all tool_call IDs from this assistant message
			expectedIDs := make(map[string]bool, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				expectedIDs[tc.ID] = false
			}

			// Look ahead for matching tool responses
			for j := i + 1; j < len(messages); j++ {
				if messages[j].Role == "tool" {
					if _, ok := expectedIDs[messages[j].ToolID]; ok {
						expectedIDs[messages[j].ToolID] = true
					}
				} else {
					break // Stop at first non-tool message
				}
			}

			// Check if all tool_calls have responses
			allFound := true
			for _, found := range expectedIDs {
				if !found {
					allFound = false
					break
				}
			}

			if !allFound {
				// Incomplete tool sequence: strip tool_calls, keep as plain text
				cleaned := m
				cleaned.ToolCalls = nil
				if strings.TrimSpace(cleaned.Content) != "" {
					out = append(out, cleaned)
				}
				continue
			}
		}

		// Remove orphan tool messages (no matching assistant tool_call above)
		if m.Role == "tool" && m.ToolID != "" {
			found := false
			for j := len(out) - 1; j >= 0; j-- {
				if out[j].Role == "assistant" && len(out[j].ToolCalls) > 0 {
					for _, tc := range out[j].ToolCalls {
						if tc.ID == m.ToolID {
							found = true
							break
						}
					}
					break
				}
				if out[j].Role != "tool" {
					break
				}
			}
			if !found {
				continue // Skip orphan tool message
			}
		}

		// Inject a synthetic assistant message between tool and user/assistant
		// Some providers (vLLM) require: assistant(tool_calls) -> tool -> assistant -> user
		if m.Role != "tool" && m.Role != "assistant" && len(out) > 0 && out[len(out)-1].Role == "tool" {
			out = append(out, Message{Role: "assistant", Content: "..."})
		}

		out = append(out, m)
	}

	return out
}

// request is the body of a chat/completions request, without "stream".
func (p *openaiCompatProvider) request(messages []Message, tools []Tool) map[string]any {
	body := map[string]any{
		"model":    p.model,
		"messages": toOpenAIMessages(messages, p.audioFormat),
	}
	if len(tools) > 0 {
		body["tools"] = tools
	}
	for k, v := range p.extraBody {
		if k == "tool_choice" && len(tools) == 0 {
			continue
		}
		body[k] = v
	}
	if p.bodyFunc != nil {
		maps.Copy(body, p.bodyFunc())
	}
	return body
}

func (p *openaiCompatProvider) Chat(ctx context.Context, messages []Message, tools []Tool) (*Message, error) {
	reqBody := p.request(messages, tools)

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("%s: marshal error: %w", p.name, err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%s: request error: %w", p.name, err)
	}
	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	p.applyExtraHeaders(req)

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("%s error %d: %s", p.name, resp.StatusCode, body)
	}

	var result struct {
		Choices []struct {
			Message Message `json:"message"`
		} `json:"choices"`
		Usage *openaiUsage `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("%s: no response", p.name)
	}

	msg := &result.Choices[0].Message
	msg.Usage = result.Usage.usage()
	return msg, nil
}

func (p *openaiCompatProvider) Stream(ctx context.Context, messages []Message, tools []Tool, onChunk func(string) error) (*Message, error) {
	reqBody := p.request(messages, tools)
	reqBody["stream"] = true
	if p.streamUsage {
		reqBody["stream_options"] = map[string]any{"include_usage": true}
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("%s: marshal error: %w", p.name, err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%s: request error: %w", p.name, err)
	}
	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	p.applyExtraHeaders(req)

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("%s stream error %d: %s", p.name, resp.StatusCode, body)
	}

	reader := bufio.NewReader(resp.Body)
	fullMessage := &Message{Role: "assistant"}
	toolCallsMap := make(map[int]*ToolCall)

	for {
		select {
		case <-ctx.Done():
			return fullMessage, ctx.Err()
		default:
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
		if data == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
					Reasoning        string `json:"reasoning"`
					ToolCalls        []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *openaiUsage `json:"usage"`
		}

		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if chunk.Usage != nil {
			fullMessage.Usage = chunk.Usage.usage()
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		delta := chunk.Choices[0].Delta

		if p.onReasoning != nil {
			if r := delta.ReasoningContent + delta.Reasoning; r != "" {
				p.onReasoning(r)
			}
		}

		if delta.Content != "" {
			fullMessage.Content += delta.Content
			if onChunk != nil {
				if err := onChunk(delta.Content); err != nil {
					return fullMessage, err
				}
			}
		}

		for _, tc := range delta.ToolCalls {
			if _, exists := toolCallsMap[tc.Index]; !exists {
				toolCallsMap[tc.Index] = &ToolCall{
					ID:       tc.ID,
					Type:     "function",
					Function: FunctionCall{Name: tc.Function.Name},
				}
			}

			currentArgs := string(toolCallsMap[tc.Index].Function.Arguments)
			toolCallsMap[tc.Index].Function.Arguments = json.RawMessage(currentArgs + tc.Function.Arguments)

			if tc.Function.Name != "" && toolCallsMap[tc.Index].Function.Name == "" {
				toolCallsMap[tc.Index].Function.Name = tc.Function.Name
			}
		}
	}

	// In the order of their index, not the random order of the map.
	for _, i := range slices.Sorted(maps.Keys(toolCallsMap)) {
		fullMessage.ToolCalls = append(fullMessage.ToolCalls, *toolCallsMap[i])
	}

	return fullMessage, nil
}
