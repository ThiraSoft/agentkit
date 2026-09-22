//go:build integration

package agentkit

import (
	"context"
	"encoding/json"
	"os"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ThiraSoft/agentkit/llm"
	"github.com/ThiraSoft/agentkit/mcp"
)

func runToolScenario(t *testing.T, cfg Config) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	var asked atomic.Bool
	secret := Tool{
		Name:        "secret_code",
		Description: "Gives the secret code of the day. No arguments.",
		Parameters: llm.ToolParams{
			Type: "object",
			Properties: llm.ToolProperties{
				"day": {Type: "string", Description: "optional"},
			},
		},
		Run: func(context.Context, json.RawMessage) (string, error) {
			asked.Store(true)
			return "PAPAYA-42", nil
		},
	}
	cfg.Tools = append(cfg.Tools, secret)
	cfg.MCP = append(cfg.MCP, mcp.ServerConfig{Name: "system", Transport: "stdio", Command: testServer(t)})

	a, err := New(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	conv := a.NewConversation("You use your tools whenever they can answer. You answer in one sentence.")

	turn, err := conv.Send(ctx, "What is the secret code of the day?", Hooks{})
	if err != nil {
		t.Fatal(err)
	}
	if !asked.Load() || !strings.Contains(turn.Text, "PAPAYA-42") {
		t.Fatalf("turn %+v, tool asked %v", turn, asked.Load())
	}

	var used []string
	_, err = conv.Send(ctx, "What time is it in Tokyo? Use the get_time tool.", Hooks{
		OnToolCall: func(c ToolCall) { used = append(used, c.Name) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(used, "get_time") {
		t.Fatalf("tools used %v, want get_time", used)
	}
}

func TestOpenAICompat(t *testing.T) {
	url := os.Getenv("AGENTKIT_OPENAI_URL")
	if url == "" {
		t.Skip("AGENTKIT_OPENAI_URL missing")
	}
	model := os.Getenv("AGENTKIT_OPENAI_MODEL")
	if model == "" {
		model = "local"
	}
	cfg := Config{
		Provider: string(llm.ProviderOpenAICompat),
		BaseURL:  url,
		Model:    model,
		APIKey:   os.Getenv("AGENTKIT_OPENAI_KEY"),
	}
	runToolScenario(t, cfg)
}
