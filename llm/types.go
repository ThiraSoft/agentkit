// types.go: core types shared by every provider: messages, tools, media,
// the Provider interface and its Config.

package llm

import (
	"context"
	"encoding/json"
)

type (
	// ProviderType identifies an LLM provider implementation.
	ProviderType string
	// MediaType identifies the kind of media attached to a message.
	MediaType string
	// Media represents binary content such as audio or an image attached to a message.
	Media struct {
		Type     MediaType `json:"type"`
		MimeType string    `json:"mime_type"` // "audio/wav", "image/jpeg", etc.
		Data     []byte    `json:"data"`      // raw bytes (base64 in JSON)
	}

	// Message is a universal message compatible with all providers.
	Message struct {
		Role      string     `json:"role"` // user, assistant, system, tool
		Content   string     `json:"content"`
		ToolCalls []ToolCall `json:"tool_calls,omitempty"`
		ToolID    string     `json:"tool_call_id,omitempty"` // for tool responses
		Name      string     `json:"name,omitempty"`         // tool name for responses

		Media []Media `json:"media,omitempty"` // for messages containing media

		// Usage is set by a provider on the message it returns; nil when
		// the provider does not report it. It is not part of the JSON.
		Usage *Usage `json:"-"`
	}
)

const (
	ProviderOpenAI       ProviderType = "openai"
	ProviderGemini       ProviderType = "gemini"
	ProviderOllama       ProviderType = "ollama"
	ProviderAnthropic    ProviderType = "anthropic"
	ProviderMistral      ProviderType = "mistral"
	ProviderLlamaCpp     ProviderType = "llamacpp"      // llama.cpp local server
	ProviderOpenAICompat ProviderType = "openai-compat" // any OpenAI-compatible server; BaseURL required

	MediaAudio MediaType = "audio"
	MediaImage MediaType = "image"
)

// ToolCall represents a tool invocation requested by the model.
type ToolCall struct {
	ID               string       `json:"id,omitempty"`
	Type             string       `json:"type,omitempty"` // "function"
	Function         FunctionCall `json:"function"`
	ThoughtSignature string       `json:"thoughtSignature,omitempty"` // for Gemini
}

// FunctionCall represents the function name and arguments of a tool call.
type FunctionCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// Tool describes a function tool that an LLM can call.
type Tool struct {
	Type     string      `json:"type"` // "function"
	Function FunctionDef `json:"function"`
}

// ToolParams defines the parameters schema for a tool.
type ToolParams struct {
	Type       string         `json:"type"` // "object"
	Properties ToolProperties `json:"properties"`
	Required   []string       `json:"required,omitempty"`
}

// ToolProperties maps argument names to their schema definitions.
type ToolProperties map[string]ToolArg

// ToolArg describes a single argument in a tool schema.
type ToolArg struct {
	Type        string `json:"type"` // JSON Schema type: "string", "number", "object", etc.
	Description string `json:"description"`
	Example     string `json:"example,omitempty"`
	Default     string `json:"default,omitempty"`
	Items       any    `json:"items,omitempty"` // for arrays
}

// FunctionDef defines the name, description and parameters of a tool
// function. Schema, when set, is a raw JSON Schema of the arguments that
// replaces Parameters for every provider. A provider written outside this
// package should send JSONSchema(), or the encoded FunctionDef, rather
// than read Parameters, which is empty when only Schema is set.
type FunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  ToolParams      `json:"parameters"`
	Schema      json.RawMessage `json:"-"`
}

// JSONSchema returns the schema of the arguments: Schema when set,
// Parameters encoded otherwise.
func (d FunctionDef) JSONSchema() json.RawMessage {
	if len(d.Schema) > 0 {
		return d.Schema
	}
	b, _ := json.Marshal(d.Parameters) // a ToolParams always encodes
	return b
}

// MarshalJSON writes JSONSchema under "parameters", the OpenAI shape.
func (d FunctionDef) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	}{d.Name, d.Description, d.JSONSchema()})
}

// UnmarshalJSON reads what MarshalJSON writes: "parameters" is kept whole
// in Schema, and fills Parameters as far as it fits.
func (d *FunctionDef) UnmarshalJSON(b []byte) error {
	var raw struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	*d = FunctionDef{Name: raw.Name, Description: raw.Description}
	if len(raw.Parameters) == 0 || string(raw.Parameters) == "null" {
		return nil
	}
	d.Schema = raw.Parameters
	// Best effort: a schema richer than ToolParams stays whole in Schema.
	_ = json.Unmarshal(raw.Parameters, &d.Parameters)
	return nil
}

// Provider is the interface for all LLMs.
type Provider interface {
	Chat(ctx context.Context, messages []Message, tools []Tool) (*Message, error)
	Stream(ctx context.Context, messages []Message, tools []Tool, onChunk func(string) error) (*Message, error)
	Name() string
	ModelName() string
}

// Config describes a provider for NewProvider.
type Config struct {
	Provider         ProviderType
	Model            string
	APIKey           string // empty = the provider's environment variable
	BaseURL          string // empty = the provider's default URL; required by "openai-compat"
	OllamaNumCtx     int    // Ollama context window, 0 = Ollama's default
	OllamaNumPredict int    // Ollama max generated tokens, 0 = MaxTokens
	// Temperature, when set, is sent as the sampling temperature. Some
	// models refuse it (recent Anthropic ones, OpenAI reasoning models).
	Temperature *float64
	// MaxTokens caps the tokens generated by one model call; 0 leaves the
	// provider's default (32000 for Anthropic, which requires a value).
	MaxTokens int
	// PromptCache marks the prompt for caching where the provider needs to
	// be told (Anthropic: the system prompt and the tools, and the
	// conversation so far). OpenAI and Gemini cache on their own.
	PromptCache bool
	// ResponseSchema, when set, is a JSON Schema the model's answers must
	// follow. Not every model takes it together with tools.
	ResponseSchema json.RawMessage
	// ExtraBody adds raw fields to every request body of the providers that
	// speak OpenAI's format (openai, mistral, llamacpp, openai-compat), after
	// the ones above, which it can override: top_p, presence_penalty,
	// chat_template_kwargs... The other providers ignore it.
	ExtraBody map[string]any
}

// Usage counts the tokens of one model call as the provider reports them;
// a count the provider does not report stays zero. InputTokens counts the
// whole prompt, the part read from or written to a cache included.
type Usage struct {
	InputTokens      int
	OutputTokens     int
	CacheReadTokens  int // prompt tokens read from the provider's cache
	CacheWriteTokens int // prompt tokens written to it (Anthropic)
}

// Add returns u plus v, count by count.
func (u Usage) Add(v Usage) Usage {
	return Usage{
		InputTokens:      u.InputTokens + v.InputTokens,
		OutputTokens:     u.OutputTokens + v.OutputTokens,
		CacheReadTokens:  u.CacheReadTokens + v.CacheReadTokens,
		CacheWriteTokens: u.CacheWriteTokens + v.CacheWriteTokens,
	}
}
