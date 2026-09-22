// openai.go: OpenAI provider (wrapper around openai_compat).

package llm

import "os"

func newOpenAIProvider(model string) *openaiCompatProvider {
	return newOpenAICompatProvider(OpenAICompatConfig{
		APIKey:  os.Getenv("OPENAI_API_KEY"),
		BaseURL: "https://api.openai.com/v1",
		Model:   model,
		Name:    "openai",
	})
}
