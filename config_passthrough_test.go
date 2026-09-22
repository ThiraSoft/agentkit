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
