package llm

import "fmt"

// NewProvider builds the provider described by cfg, wrapped with WithRetry.
// APIKey and BaseURL, when set, take precedence over the environment
// variable and the default URL of the provider. "openai-compat" requires
// BaseURL. An unknown provider is an error.
func NewProvider(cfg Config) (Provider, error) {
	var provider Provider
	switch cfg.Provider {
	case ProviderOpenAI:
		provider = newOpenAIProvider(cfg.Model)
	case ProviderGemini:
		provider = newGeminiProvider(cfg.Model)
	case ProviderOllama:
		provider = newOllamaProvider(cfg.Model, cfg.OllamaNumCtx, cfg.OllamaNumPredict)
	case ProviderAnthropic:
		provider = newAnthropicProvider(cfg.Model)
	case ProviderMistral:
		provider = newMistralProvider(cfg.Model)
	case ProviderLlamaCpp:
		provider = newLlamaCppProvider(cfg.Model)
	case ProviderOpenAICompat:
		if cfg.BaseURL == "" {
			return nil, fmt.Errorf("openai-compat: BaseURL is required")
		}
		provider = newOpenAICompatProvider(OpenAICompatConfig{
			BaseURL:   cfg.BaseURL,
			Model:     cfg.Model,
			Name:      string(ProviderOpenAICompat),
			ExtraBody: map[string]any{"tool_choice": "auto"},
		})
	default:
		return nil, fmt.Errorf("unknown provider: %s", cfg.Provider)
	}

	applyOverrides(provider, cfg)
	return WithRetry(provider), nil
}

// applyOverrides sets APIKey and BaseURL from cfg on providers that support them.
func applyOverrides(p Provider, cfg Config) {
	switch v := p.(type) {
	case *geminiProvider:
		if cfg.APIKey != "" {
			v.apiKey = cfg.APIKey
		}
		if cfg.BaseURL != "" {
			v.baseURL = cfg.BaseURL
		}
	case *anthropicProvider:
		if cfg.APIKey != "" {
			v.apiKey = cfg.APIKey
		}
		if cfg.BaseURL != "" {
			v.baseURL = cfg.BaseURL
		}
	case *openaiCompatProvider:
		if cfg.APIKey != "" {
			v.apiKey = cfg.APIKey
		}
		if cfg.BaseURL != "" {
			v.baseURL = cfg.BaseURL
		}
	case *ollamaProvider:
		if cfg.BaseURL != "" {
			v.baseURL = cfg.BaseURL
		}
	}
}
