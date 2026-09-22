// gemini.go: the Gemini provider.

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
	"os"
	"strings"
	"time"
)

type geminiProvider struct {
	baseURL     string
	apiKey      string
	Model       string
	client      *http.Client
	temperature *float64
	maxTokens   int
}

func newGeminiProvider(cfg Config) *geminiProvider {
	return &geminiProvider{
		baseURL:     or(cfg.BaseURL, "https://generativelanguage.googleapis.com/v1beta"),
		apiKey:      or(cfg.APIKey, os.Getenv("GEMINI_API_KEY")),
		Model:       cfg.Model,
		client:      &http.Client{Timeout: 120 * time.Second},
		temperature: cfg.Temperature,
		maxTokens:   cfg.MaxTokens,
	}
}

// generationConfig is the generationConfig of a request, nil when there
// is nothing to set.
func (p *geminiProvider) generationConfig() map[string]any {
	gc := map[string]any{}
	if p.temperature != nil {
		gc["temperature"] = *p.temperature
	}
	if p.maxTokens > 0 {
		gc["maxOutputTokens"] = p.maxTokens
	}
	if len(gc) == 0 {
		return nil
	}
	return gc
}

// request is the body of a generateContent request.
func (p *geminiProvider) request(messages []Message, tools []Tool) map[string]any {
	systemPrompt, contents := p.convertMessages(messages)
	body := map[string]any{
		"contents": contents,
	}
	if systemPrompt != "" {
		body["systemInstruction"] = map[string]any{
			"parts": []map[string]any{
				{"text": systemPrompt},
			},
		}
	}
	if len(tools) > 0 {
		body["tools"] = []map[string]any{
			{"function_declarations": p.convertTools(tools)},
		}
	}
	if gc := p.generationConfig(); gc != nil {
		body["generationConfig"] = gc
	}
	return body
}

func (p *geminiProvider) ModelName() string { return p.Model }

func (p *geminiProvider) Name() string { return "gemini" }

// geminiUsage is the usageMetadata of a Gemini answer; in a stream each
// chunk carries the counts so far.
type geminiUsage struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
}

func (u *geminiUsage) usage() *Usage {
	if u == nil {
		return nil
	}
	return &Usage{
		InputTokens:     u.PromptTokenCount,
		OutputTokens:    u.CandidatesTokenCount + u.ThoughtsTokenCount,
		CacheReadTokens: u.CachedContentTokenCount,
	}
}

func (p *geminiProvider) Chat(ctx context.Context, messages []Message, tools []Tool) (*Message, error) {
	reqBody := p.request(messages, tools)

	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", p.baseURL, p.Model, p.apiKey)
	body, _ := json.Marshal(reqBody)
	req, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini error %d: %s", resp.StatusCode, body)
	}

	var result struct {
		Candidates []struct {
			Content struct {
				Parts []map[string]any `json:"parts"`
				Role  string           `json:"role"`
			} `json:"content"`
		} `json:"candidates"`
		UsageMetadata *geminiUsage `json:"usageMetadata"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if len(result.Candidates) == 0 {
		return nil, fmt.Errorf("no response")
	}

	msg := p.convertResponse(result.Candidates[0].Content)
	msg.Usage = result.UsageMetadata.usage()
	return msg, nil
}

// Stream implements the Provider interface
func (p *geminiProvider) Stream(ctx context.Context, messages []Message, tools []Tool, onChunk func(string) error) (*Message, error) {
	reqBody := p.request(messages, tools)

	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?key=%s&alt=sse", p.baseURL, p.Model, p.apiKey)

	bodyJSON, _ := json.Marshal(reqBody)
	req, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyJSON))
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini stream error %d: %s", resp.StatusCode, body)
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
			if ctx.Err() != nil {
				return fullMessage, ctx.Err()
			}
			return fullMessage, err
		}

		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		var chunk struct {
			Candidates []struct {
				Content struct {
					Parts []map[string]any `json:"parts"`
					Role  string           `json:"role"`
				} `json:"content"`
			} `json:"candidates"`
			UsageMetadata *geminiUsage `json:"usageMetadata"`
		}

		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if chunk.UsageMetadata != nil {
			fullMessage.Usage = chunk.UsageMetadata.usage()
		}

		if len(chunk.Candidates) > 0 {
			partMsg := p.convertResponse(chunk.Candidates[0].Content)

			if partMsg.Content != "" {
				if onChunk != nil {
					if err := onChunk(partMsg.Content); err != nil {
						return fullMessage, err
					}
				}
				fullMessage.Content += partMsg.Content
			}

			if len(partMsg.ToolCalls) > 0 {
				fullMessage.ToolCalls = append(fullMessage.ToolCalls, partMsg.ToolCalls...)
			}
		}
	}

	return fullMessage, nil
}

func (p *geminiProvider) convertMessages(messages []Message) (string, []map[string]any) {
	var systemPrompt string
	var contents []map[string]any

	for i := 0; i < len(messages); i++ {
		msg := messages[i]

		// Extract system prompt separately
		if msg.Role == "system" {
			systemPrompt += msg.Content + "\n"
			continue
		}

		role := msg.Role
		if role == "assistant" {
			role = "model"
		}

		parts := []map[string]any{}

		// Media
		for _, m := range msg.Media {
			parts = append(parts, map[string]any{
				"inline_data": map[string]any{
					"mime_type": m.MimeType,
					"data":      base64.StdEncoding.EncodeToString(m.Data),
				},
			})
		}

		// Tool responses: group all consecutive tool results
		if msg.Role == "tool" {
			toolParts := []map[string]any{}
			for i < len(messages) && messages[i].Role == "tool" {
				toolParts = append(toolParts, map[string]any{
					"functionResponse": map[string]any{
						"name":     messages[i].Name,
						"response": map[string]any{"result": messages[i].Content},
					},
				})
				i++
			}
			i-- // Compensate for the main loop's i++

			contents = append(contents, map[string]any{
				"role":  "user",
				"parts": toolParts,
			})
			continue
		}

		// Regular text
		if msg.Content != "" {
			parts = append(parts, map[string]any{"text": msg.Content})
		}

		// Model tool calls
		for _, tc := range msg.ToolCalls {
			var args map[string]any
			json.Unmarshal(tc.Function.Arguments, &args)

			fcPart := map[string]any{
				"functionCall": map[string]any{
					"name": tc.Function.Name,
					"args": args,
				},
			}

			if tc.ThoughtSignature != "" {
				fcPart["thoughtSignature"] = tc.ThoughtSignature
			}

			parts = append(parts, fcPart)
		}

		contents = append(contents, map[string]any{
			"role":  role,
			"parts": parts,
		})
	}

	return systemPrompt, contents
}

func (p *geminiProvider) convertResponse(content struct {
	Parts []map[string]any `json:"parts"`
	Role  string           `json:"role"`
},
) *Message {
	msg := &Message{Role: "assistant"}

	for _, part := range content.Parts {
		if text, ok := part["text"].(string); ok {
			msg.Content += text
		}

		if fc, ok := part["functionCall"].(map[string]any); ok {
			name := fc["name"].(string)
			args, _ := json.Marshal(fc["args"])

			thoughtSig := ""
			if ts, ok := part["thoughtSignature"].(string); ok {
				thoughtSig = ts
			}

			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:               "", // Leave empty so the server generates a unique ID
				Type:             "function",
				ThoughtSignature: thoughtSig,
				Function: FunctionCall{
					Name:      name,
					Arguments: args,
				},
			})
		}
	}

	return msg
}

func (p *geminiProvider) convertTools(tools []Tool) []map[string]any {
	var defs []map[string]any
	for _, t := range tools {
		def := map[string]any{
			"name":        t.Function.Name,
			"description": t.Function.Description,
		}
		if len(t.Function.Schema) > 0 {
			def["parametersJsonSchema"] = withoutMetaSchema(t.Function.Schema)
		} else {
			// Sanitize parameters to ensure Gemini compatibility
			def["parameters"] = p.sanitizeParameters(t.Function.Parameters)
		}
		defs = append(defs, def)
	}
	return defs
}

// withoutMetaSchema drops the top-level "$schema" key of a JSON Schema,
// which Gemini does not take.
func withoutMetaSchema(schema json.RawMessage) any {
	var m map[string]any
	if err := json.Unmarshal(schema, &m); err != nil {
		return schema
	}
	delete(m, "$schema")
	return m
}

// sanitizeParameters ensures the tool parameters are valid for Gemini's JSON Schema requirements.
// Gemini requires that arrays have an "items" field defined.
func (p *geminiProvider) sanitizeParameters(params ToolParams) map[string]any {
	result := map[string]any{
		"type": params.Type,
	}

	if len(params.Required) > 0 {
		result["required"] = params.Required
	}

	if len(params.Properties) > 0 {
		props := make(map[string]any)
		for name, arg := range params.Properties {
			props[name] = p.sanitizeProperty(arg)
		}
		result["properties"] = props
	}

	return result
}

// sanitizeProperty ensures a single property is valid for Gemini.
// For arrays, it ensures "items" is always present.
func (p *geminiProvider) sanitizeProperty(arg ToolArg) map[string]any {
	prop := map[string]any{
		"type": arg.Type,
	}

	if arg.Description != "" {
		prop["description"] = arg.Description
	}
	if arg.Example != "" {
		prop["example"] = arg.Example
	}
	if arg.Default != "" {
		prop["default"] = arg.Default
	}

	// For arrays, ensure items is always present
	if arg.Type == "array" {
		if arg.Items != nil {
			// Recursively sanitize nested items if it's a map
			if itemsMap, ok := arg.Items.(map[string]any); ok {
				prop["items"] = p.sanitizeItemsSchema(itemsMap)
			} else {
				prop["items"] = arg.Items
			}
		} else {
			// Default to string items if not specified
			prop["items"] = map[string]any{"type": "string"}
		}
	}

	return prop
}

// sanitizeItemsSchema recursively sanitizes nested schemas in items.
func (p *geminiProvider) sanitizeItemsSchema(items map[string]any) map[string]any {
	result := make(map[string]any)

	for k, v := range items {
		if k == "properties" {
			// Handle nested properties
			if propsMap, ok := v.(map[string]any); ok {
				sanitizedProps := make(map[string]any)
				for propName, propVal := range propsMap {
					if propArg, ok := propVal.(ToolArg); ok {
						sanitizedProps[propName] = p.sanitizeProperty(propArg)
					} else if propMap, ok := propVal.(map[string]any); ok {
						// Already a map, check if it's an array needing items
						sanitizedProps[propName] = p.sanitizeMapProperty(propMap)
					} else {
						sanitizedProps[propName] = propVal
					}
				}
				result[k] = sanitizedProps
			} else if propsToolProps, ok := v.(ToolProperties); ok {
				// Handle ToolProperties type
				sanitizedProps := make(map[string]any)
				for propName, propArg := range propsToolProps {
					sanitizedProps[propName] = p.sanitizeProperty(propArg)
				}
				result[k] = sanitizedProps
			} else {
				result[k] = v
			}
		} else if k == "items" {
			// Recursively sanitize nested items
			if nestedItems, ok := v.(map[string]any); ok {
				result[k] = p.sanitizeItemsSchema(nestedItems)
			} else {
				result[k] = v
			}
		} else {
			result[k] = v
		}
	}

	// If this is an array type without items, add default
	if result["type"] == "array" && result["items"] == nil {
		result["items"] = map[string]any{"type": "string"}
	}

	return result
}

// sanitizeMapProperty ensures a map-based property definition is valid for Gemini.
func (p *geminiProvider) sanitizeMapProperty(prop map[string]any) map[string]any {
	result := make(map[string]any)
	for k, v := range prop {
		result[k] = v
	}

	// If it's an array without items, add default items
	if result["type"] == "array" && result["items"] == nil {
		result["items"] = map[string]any{"type": "string"}
	}

	return result
}
