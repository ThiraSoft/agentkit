package llm

import (
	"fmt"
	"time"
)

// NewProvider builds the provider described by cfg, wrapped with WithRetry.
// APIKey and BaseURL, when set, take precedence over the environment
// variable and the default URL of the provider. "openai-compat" requires
// BaseURL. An unknown provider is an error.
func NewProvider(cfg Config) (Provider, error) {
	var provider Provider
	switch cfg.Provider {
	case ProviderOpenAI:
		provider = newOpenAIProvider(cfg)
	case ProviderGemini:
		provider = newGeminiProvider(cfg)
	case ProviderOllama:
		provider = newOllamaProvider(cfg)
	case ProviderAnthropic:
		provider = newAnthropicProvider(cfg)
	case ProviderMistral:
		provider = newMistralProvider(cfg)
	case ProviderLlamaCpp:
		provider = newLlamaCppProvider(cfg)
	case ProviderOpenAICompat:
		if cfg.BaseURL == "" {
			return nil, fmt.Errorf("openai-compat: BaseURL is required")
		}
		provider = newOpenAICompatProvider(OpenAICompatConfig{
			APIKey:        cfg.APIKey,
			BaseURL:       cfg.BaseURL,
			Model:         cfg.Model,
			Name:          string(ProviderOpenAICompat),
			ExtraBody:     requestFields(cfg, "max_tokens", map[string]any{"tool_choice": "auto"}),
			StreamUsage:   true,
			Timeout:       cfg.Timeout,
			DropReasoning: cfg.DropReasoning,
		})
	default:
		return nil, fmt.Errorf("unknown provider: %s", cfg.Provider)
	}
	return WithRetry(provider), nil
}

// timeout is cfg.Timeout, or def when it is not set.
func timeout(cfg Config, def time.Duration) time.Duration {
	if cfg.Timeout > 0 {
		return cfg.Timeout
	}
	return def
}

// or returns s, or def when s is empty.
func or(s, def string) string {
	if s != "" {
		return s
	}
	return def
}

// requestFields adds to fields, a new map when nil, the body fields of an
// OpenAI-compatible request that cfg asks for: the temperature, the token
// cap under maxField, the response format, and ExtraBody last.
func requestFields(cfg Config, maxField string, fields map[string]any) map[string]any {
	if fields == nil {
		fields = map[string]any{}
	}
	if cfg.Temperature != nil {
		fields["temperature"] = *cfg.Temperature
	}
	if cfg.MaxTokens > 0 {
		fields[maxField] = cfg.MaxTokens
	}
	if len(cfg.ResponseSchema) > 0 {
		fields["response_format"] = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "response",
				"schema": cfg.ResponseSchema,
			},
		}
	}
	for k, v := range cfg.ExtraBody {
		fields[k] = v
	}
	return fields
}
