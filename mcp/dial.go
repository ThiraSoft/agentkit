package mcp

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"
)

// envVarPattern matches ${VAR_NAME} placeholders for env-var substitution.
var envVarPattern = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)\}`)

// expandEnv replaces every ${VAR} in s with os.Getenv("VAR"). Unknown vars
// expand to the empty string, matching standard shell behavior.
func expandEnv(s string) string {
	return envVarPattern.ReplaceAllStringFunc(s, func(match string) string {
		return os.Getenv(match[2 : len(match)-1])
	})
}

// expanded returns a copy of cfg with ${VAR} placeholders replaced in URL,
// Command, and header values. The stored config keeps the raw placeholders.
func (c ServerConfig) expanded() ServerConfig {
	out := c
	out.URL = expandEnv(c.URL)
	out.Command = expandEnv(c.Command)
	if len(c.Headers) > 0 {
		out.Headers = make(map[string]string, len(c.Headers))
		for k, v := range c.Headers {
			out.Headers[k] = expandEnv(v)
		}
	}
	return out
}

// Dial connects to the server described by cfg and fetches its tools.
// ${VAR} placeholders in Command, URL and header values are replaced with
// environment variables first. Fetching is tried three times, 500 ms
// apart, since a stdio server may still be starting. The caller closes the
// returned client.
func Dial(ctx context.Context, cfg ServerConfig) (*Client, error) {
	cfg = cfg.expanded()
	var (
		client *Client
		err    error
	)
	switch cfg.Transport {
	case "stdio":
		full := strings.Split(cfg.Command, " ")
		client, err = NewStdioClient(ctx, full[0], full[1:]...)
	case "sse":
		client, err = NewSSEClient(ctx, cfg.URL, cfg.Headers)
	case "streamable":
		client, err = NewStreamableClient(ctx, cfg.URL, cfg.Headers)
	default:
		return nil, fmt.Errorf("unknown transport: %s", cfg.Transport)
	}
	if err != nil {
		return nil, err
	}
	for attempt := range 3 {
		if err = client.FetchTools(ctx); err == nil {
			return client, nil
		}
		if attempt < 2 {
			timer := time.NewTimer(500 * time.Millisecond)
			select {
			case <-ctx.Done():
				timer.Stop()
				client.Close()
				return nil, ctx.Err()
			case <-timer.C:
			}
		}
	}
	client.Close()
	return nil, err
}
