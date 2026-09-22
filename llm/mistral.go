// mistral.go: Mistral provider (wrapper around openai_compat).

package llm

import "os"

func newMistralProvider(model string) *openaiCompatProvider {
	return newOpenAICompatProvider(OpenAICompatConfig{
		APIKey:  os.Getenv("MISTRAL_API_KEY"),
		BaseURL: "https://api.mistral.ai/v1",
		Model:   model,
		Name:    "mistral",
		ExtraBody: map[string]any{
			"tool_choice": "auto",
		},
	})
}
