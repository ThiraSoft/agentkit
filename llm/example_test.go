package llm_test

import (
	"fmt"
	"log"

	"github.com/ThiraSoft/agentkit/llm"
)

func ExampleNewProvider() {
	p, err := llm.NewProvider(llm.Config{
		Provider: llm.ProviderOpenAICompat,
		Model:    "local-model",
		BaseURL:  "http://localhost:8080/v1",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(p.Name(), p.ModelName())
	// Output: openai-compat local-model
}

// A provider for an OpenAI-compatible server that needs its own settings,
// with the same retries as NewProvider.
func ExampleNewOpenAICompat() {
	p := llm.WithRetry(llm.NewOpenAICompat(llm.OpenAICompatConfig{
		BaseURL:   "http://localhost:8000/v1",
		Model:     "local-model",
		Name:      "local",
		ExtraBody: map[string]any{"temperature": 0.2},
	}))
	fmt.Println(p.Name(), p.ModelName())
	// Output: local local-model
}
