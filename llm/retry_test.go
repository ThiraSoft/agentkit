package llm

import (
	"context"
	"errors"
	"testing"
)

// flakyProvider fails n times then responds.
type flakyProvider struct {
	failures int
	calls    int
	err      error
}

func (f *flakyProvider) Chat(_ context.Context, _ []Message, _ []Tool) (*Message, error) {
	f.calls++
	if f.calls <= f.failures {
		return nil, f.err
	}
	return &Message{Role: "assistant", Content: "ok"}, nil
}

func (f *flakyProvider) Stream(ctx context.Context, msgs []Message, tools []Tool, onChunk func(string) error) (*Message, error) {
	resp, err := f.Chat(ctx, msgs, tools)
	if err != nil {
		return nil, err
	}
	_ = onChunk(resp.Content)
	return resp, nil
}

func (f *flakyProvider) Name() string      { return "flaky" }
func (f *flakyProvider) ModelName() string { return "flaky-model" }

func TestRetry_TransientErrorRecovered(t *testing.T) {
	inner := &flakyProvider{failures: 2, err: errors.New("429 too many requests")}
	p := WithRetry(inner)

	resp, err := p.Chat(context.Background(), nil, nil)
	if err != nil {
		t.Fatalf("expected recovery, got %v", err)
	}
	if resp.Content != "ok" || inner.calls != 3 {
		t.Errorf("unexpected: %+v calls=%d", resp, inner.calls)
	}
}

func TestRetry_NonTransientFailsFast(t *testing.T) {
	inner := &flakyProvider{failures: 10, err: errors.New("invalid api key")}
	p := WithRetry(inner)

	if _, err := p.Chat(context.Background(), nil, nil); err == nil {
		t.Fatal("expected error")
	}
	if inner.calls != 1 {
		t.Errorf("non-transient error must not retry, got %d calls", inner.calls)
	}
}

func TestRetry_GivesUpAfterMaxAttempts(t *testing.T) {
	inner := &flakyProvider{failures: 10, err: errors.New("503 service unavailable")}
	p := WithRetry(inner)

	if _, err := p.Chat(context.Background(), nil, nil); err == nil {
		t.Fatal("expected error after max attempts")
	}
	if inner.calls != retryMaxAttempts {
		t.Errorf("expected %d attempts, got %d", retryMaxAttempts, inner.calls)
	}
}

// streamPartialProvider emits a chunk then fails, on every call.
type streamPartialProvider struct{ calls int }

func (s *streamPartialProvider) Chat(context.Context, []Message, []Tool) (*Message, error) {
	return nil, errors.New("not used")
}
func (s *streamPartialProvider) Stream(_ context.Context, _ []Message, _ []Tool, onChunk func(string) error) (*Message, error) {
	s.calls++
	_ = onChunk("partial")
	return nil, errors.New("503 service unavailable")
}
func (s *streamPartialProvider) Name() string      { return "partial" }
func (s *streamPartialProvider) ModelName() string { return "partial" }

func TestRetry_StreamNeverRetriesAfterEmission(t *testing.T) {
	inner := &streamPartialProvider{}
	p := WithRetry(inner)

	_, err := p.Stream(context.Background(), nil, nil, func(string) error { return nil })
	if err == nil {
		t.Fatal("expected error")
	}
	if inner.calls != 1 {
		t.Errorf("stream with emitted content must not retry, got %d calls", inner.calls)
	}
}

func TestRetry_ContextCanceledNotRetried(t *testing.T) {
	inner := &flakyProvider{failures: 10, err: context.Canceled}
	p := WithRetry(inner)

	_, err := p.Chat(context.Background(), nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if inner.calls != 1 {
		t.Errorf("canceled context must not retry, got %d calls", inner.calls)
	}
}

func TestWithRetryIsExportedAndIdempotent(t *testing.T) {
	inner := &flakyProvider{}
	p := WithRetry(inner)
	if WithRetry(p) != p {
		t.Fatal("WithRetry wrapped an already wrapped provider")
	}
	if UnwrapProvider(p) != Provider(inner) {
		t.Fatal("UnwrapProvider did not give back the inner provider")
	}
	if WithRetry(nil) != nil {
		t.Fatal("WithRetry(nil) is not nil")
	}
}
