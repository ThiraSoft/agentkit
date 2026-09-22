// openai.go: OpenAI provider (wrapper around openai_compat).

package llm

import "os"

func newOpenAIProvider(cfg Config) *openaiCompatProvider {
	return newOpenAICompatProvider(OpenAICompatConfig{
		APIKey:    or(cfg.APIKey, os.Getenv("OPENAI_API_KEY")),
		BaseURL:   or(cfg.BaseURL, "https://api.openai.com/v1"),
		Model:     cfg.Model,
		Name:      "openai",
		ExtraBody: requestFields(cfg, "max_completion_tokens", nil),
	})
}
