// mistral.go: Mistral provider (wrapper around openai_compat).

package llm

import "os"

func newMistralProvider(cfg Config) *openaiCompatProvider {
	return newOpenAICompatProvider(OpenAICompatConfig{
		APIKey:    or(cfg.APIKey, os.Getenv("MISTRAL_API_KEY")),
		BaseURL:   or(cfg.BaseURL, "https://api.mistral.ai/v1"),
		Model:     cfg.Model,
		Name:      "mistral",
		ExtraBody: requestFields(cfg, "max_tokens", map[string]any{"tool_choice": "auto"}),
	})
}
