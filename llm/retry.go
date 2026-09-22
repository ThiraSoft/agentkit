package llm

// retryProvider wraps a Provider with exponential backoff retry on
// transient errors (rate limit, overload, network drops). The stream
// is only retried if no chunk has been emitted yet, so we never
// duplicate content already shown.

import (
	"context"
	"errors"
	"net"
	"strings"
	"time"
)

const (
	retryMaxAttempts = 3
	retryBaseDelay   = time.Second
)

type retryProvider struct {
	Provider
}

// WithRetry wraps p so that transient errors (rate limit, overload, network
// drops) are retried with exponential backoff, up to 3 attempts. A stream is
// retried only while no chunk has been emitted, so shown text is never
// duplicated. WithRetry returns nil for nil and p itself when p is already
// wrapped. NewProvider applies it to every provider it builds; use it on a
// provider built with NewOpenAICompat or written outside this package.
func WithRetry(p Provider) Provider {
	if p == nil {
		return nil
	}
	if _, ok := p.(*retryProvider); ok {
		return p
	}
	return &retryProvider{Provider: p}
}

// UnwrapProvider removes the decorators (retry) to get back the concrete
// provider, for instance to type-assert a provider you wrote yourself.
func UnwrapProvider(p Provider) Provider {
	if r, ok := p.(*retryProvider); ok {
		return r.Provider
	}
	return p
}

// isTransient identifies errors worth retrying.
func isTransient(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	msg := strings.ToLower(err.Error())
	for _, marker := range []string{
		"429", "rate limit", "resource exhausted", "resource_exhausted",
		"500", "502", "503", "504",
		"overloaded", "unavailable", "internal server error",
		"connection reset", "connection refused", "eof",
	} {
		if strings.Contains(msg, marker) {
			return true
		}
	}
	return false
}

func backoffWait(ctx context.Context, attempt int) error {
	delay := retryBaseDelay << attempt // 1s, 2s, 4s
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(delay):
		return nil
	}
}

func (r *retryProvider) Chat(ctx context.Context, messages []Message, tools []Tool) (*Message, error) {
	var lastErr error
	for attempt := 0; attempt < retryMaxAttempts; attempt++ {
		resp, err := r.Provider.Chat(ctx, messages, tools)
		if err == nil || !isTransient(err) {
			return resp, err
		}
		lastErr = err
		if attempt < retryMaxAttempts-1 {
			if werr := backoffWait(ctx, attempt); werr != nil {
				return nil, werr
			}
		}
	}
	return nil, lastErr
}

func (r *retryProvider) Stream(ctx context.Context, messages []Message, tools []Tool, onChunk func(string) error) (*Message, error) {
	var lastErr error
	for attempt := 0; attempt < retryMaxAttempts; attempt++ {
		emitted := false
		wrapped := func(s string) error {
			if s != "" {
				emitted = true
			}
			return onChunk(s)
		}
		resp, err := r.Provider.Stream(ctx, messages, tools, wrapped)
		// No retry once content has been emitted: we never duplicate
		// what the user has already seen.
		if err == nil || emitted || !isTransient(err) {
			return resp, err
		}
		lastErr = err
		if attempt < retryMaxAttempts-1 {
			if werr := backoffWait(ctx, attempt); werr != nil {
				return nil, werr
			}
		}
	}
	return nil, lastErr
}
