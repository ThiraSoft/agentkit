package agentkit

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// passthroughServer is an OpenAI-compatible server that answers "ok" and
// keeps the JSON body of the last request.
func passthroughServer(t *testing.T) (string, func() map[string]any) {
	t.Helper()
	var mu sync.Mutex
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b map[string]any
		_ = json.NewDecoder(r.Body).Decode(&b)
		mu.Lock()
		body = b
		mu.Unlock()
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv.URL + "/v1", func() map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return body
	}
}

func TestConfigReachesTheProvider(t *testing.T) {
	url, body := passthroughServer(t)
	temp := 0.2
	a, err := New(context.Background(), Config{
		Provider:    "openai-compat",
		BaseURL:     url,
		Model:       "m",
		Temperature: &temp,
		MaxTokens:   64,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if _, err := a.NewConversation("s").Send(context.Background(), "hi", Hooks{}); err != nil {
		t.Fatal(err)
	}
	if b := body(); b["temperature"] != 0.2 || b["max_tokens"] != 64.0 {
		t.Fatalf("body %v", b)
	}
}

func TestPromptCacheReachesAnthropic(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n")
	}))
	defer srv.Close()
	a, err := New(context.Background(), Config{Provider: "anthropic", Model: "m", APIKey: "k", BaseURL: srv.URL, PromptCache: true})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if _, err := a.NewConversation("s").Send(context.Background(), "hi", Hooks{}); err != nil {
		t.Fatal(err)
	}
	if _, has := body["cache_control"]; !has {
		t.Fatalf("body %v", body)
	}
}

type verdict struct {
	Answer string `json:"answer"`
}

func TestResponseSchemaReachesTheProvider(t *testing.T) {
	url, body := passthroughServer(t)
	schema, err := SchemaFor[verdict]()
	if err != nil {
		t.Fatal(err)
	}
	a, err := New(context.Background(), Config{Provider: "openai-compat", BaseURL: url, Model: "m", ResponseSchema: schema})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if _, err := a.NewConversation("s").Send(context.Background(), "hi", Hooks{}); err != nil {
		t.Fatal(err)
	}
	rf, _ := body()["response_format"].(map[string]any)
	if rf["type"] != "json_schema" {
		t.Fatalf("body %v", body())
	}
}

func TestNewRejectsAResponseSchemaThatIsNotAnObject(t *testing.T) {
	if a, err := New(context.Background(), Config{ProviderImpl: &fakeProvider{}, ResponseSchema: json.RawMessage(`[]`)}); err == nil {
		a.Close()
		t.Fatal("ResponseSchema [] accepted")
	}
}
