//go:build integration

package agentkit

import (
	"os"
	"testing"
)

// A real turn on Gemini: a Go tool, then an MCP tool.
func TestGemini(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY missing")
	}
	runToolScenario(t, Config{
		Provider: "gemini",
		Model:    "gemini-2.5-flash-lite",
	})
}
