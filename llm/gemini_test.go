package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestGeminiStreamNetworkCutReturnsErrorAndPartialText(t *testing.T) {
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("expected http.Flusher")
			return
		}
		// Send an SSE chunk with "Début "
		fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Début \"}],\"role\":\"model\"}}]}\n\n")
		flusher.Flush()

		// Abruptly cut the TCP connection via Hijacker
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Error("expected http.Hijacker")
			return
		}
		conn, _, err := hj.Hijack()
		if err != nil {
			t.Errorf("hijack failed: %v", err)
			return
		}
		conn.Close()
	}))
	defer srv.Close()

	p, err := NewProvider(Config{
		Provider: ProviderGemini,
		Model:    "gemini-test",
		BaseURL:  srv.URL,
		APIKey:   "test-key",
	})
	if err != nil {
		t.Fatal(err)
	}

	var chunks []string
	msg, err := p.Stream(context.Background(), []Message{{Role: "user", Content: "bonjour"}}, nil, func(c string) error {
		chunks = append(chunks, c)
		return nil
	})
	if err == nil {
		t.Fatal("expected error on a network cut, got nil")
	}
	if msg == nil || msg.Content != "Début " {
		t.Fatalf("msg %+v, expected partial content 'Début '", msg)
	}
	if got := strings.Join(chunks, ""); got != "Début " {
		t.Fatalf("chunks %q, expected 'Début '", got)
	}
	if attempts.Load() != 1 {
		t.Fatalf("attempts %d, expected 1 (WithRetry must not retry after emission)", attempts.Load())
	}
}
