// openai.go: OpenAI provider (wrapper around openai_compat).

package llm

import "os"

func newOpenAIProvider(cfg Config) *openaiCompatProvider {
	return newOpenAICompatProvider(OpenAICompatConfig{
		APIKey:        or(cfg.APIKey, os.Getenv("OPENAI_API_KEY")),
		BaseURL:       or(cfg.BaseURL, "https://api.openai.com/v1"),
		Model:         cfg.Model,
		Name:          "openai",
		ExtraBody:     openAIFields(cfg),
		StreamUsage:   true,
		Timeout:       cfg.Timeout,
		DropReasoning: cfg.DropReasoning,
	})
}

// openAIFields are requestFields, with the reasoning_effort that
// Config.Thinking asks of a reasoning model.
func openAIFields(cfg Config) map[string]any {
	fields := requestFields(cfg, "max_completion_tokens", nil)
	if cfg.Thinking != nil {
		if effort := openAIEffort(cfg.Model, *cfg.Thinking); effort != "" {
			fields["reasoning_effort"] = effort
		}
	}
	return fields
}
