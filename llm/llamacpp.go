// llamacpp.go: llama.cpp provider (wrapper around openai_compat).
// llama-server exposes an OpenAI-compatible API on /v1/chat/completions.
// Default URL: http://localhost:8080. Override with LLAMACPP_URL.

package llm

import (
	"encoding/json"
	"net/http"
	"os"
	"time"
)

func llamaCppBaseURL() string {
	if url := os.Getenv("LLAMACPP_URL"); url != "" {
		return url
	}
	return "http://localhost:8080"
}

func newLlamaCppProvider(cfg Config) *openaiCompatProvider {
	return newOpenAICompatProvider(OpenAICompatConfig{
		APIKey:    cfg.APIKey,
		BaseURL:   or(cfg.BaseURL, llamaCppBaseURL()+"/v1"),
		Model:     cfg.Model,
		Name:      "llamacpp",
		ExtraBody: requestFields(cfg, "max_tokens", nil),
		// llama-server (mtmd) follows the OpenAI convention: input_audio block.
		AudioFormat: "input_audio",
		StreamUsage: true,
	})
}

// ListLlamaCppModels queries the local llama.cpp server for the list
// of loaded models. Returns nil if the server is unreachable.
func ListLlamaCppModels() []string {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(llamaCppBaseURL() + "/v1/models")
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	var result struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil
	}
	models := make([]string, 0, len(result.Data))
	for _, m := range result.Data {
		models = append(models, m.ID)
	}
	return models
}
