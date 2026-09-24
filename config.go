package agentkit

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ThiraSoft/agentkit/mcp"
)

// fileConfig is the JSON that LoadConfig reads.
type fileConfig struct {
	Provider       string             `json:"provider"`
	Model          string             `json:"model"`
	BaseURL        string             `json:"baseURL"`
	APIKey         string             `json:"apiKey"`
	MaxSteps       int                `json:"maxSteps"`
	MaxToolResult  int                `json:"maxToolResult"`
	Temperature    *float64           `json:"temperature"`
	MaxTokens      int                `json:"maxTokens"`
	PromptCache    bool               `json:"promptCache"`
	ResponseSchema json.RawMessage    `json:"responseSchema"`
	MCP            []mcp.ServerConfig `json:"mcp"`
	Tools          []string           `json:"tools"`
	Workdir        string             `json:"workdir"`
	System         string             `json:"system"`
	ExtraBody      map[string]any     `json:"extraBody"`
	Timeout        int                `json:"timeout"`
}

// placeholder matches the ${VAR} that LoadConfig expands.
var placeholder = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)\}`)

// LoadConfig reads a Config from a JSON file:
//
//	{
//	  "provider": "openai-compat",
//	  "model": "local",
//	  "baseURL": "http://localhost:8080/v1",
//	  "apiKey": "${LOCAL_KEY}",
//	  "maxSteps": 20,
//	  "maxToolResult": 32768,
//	  "temperature": 0.7,
//	  "maxTokens": 4096,
//	  "promptCache": false,
//	  "responseSchema": {"type": "object"},
//	  "extraBody": {"top_p": 0.8, "chat_template_kwargs": {"enable_thinking": true}},
//	  "timeout": 1800,
//	  "mcp": [{"name": "files", "transport": "stdio", "command": "files-mcp --root /tmp"}],
//	  "tools": ["read_file", "edit_file", "grep"],
//	  "workdir": "~/src/project",
//	  "system": "You are a careful engineer."
//	}
//
// timeout is in seconds, the longest one model call may take. Only provider is required. tools names built-in tools (Config.Builtins),
// none if absent. ${VAR} in apiKey, baseURL and workdir is replaced by the
// environment variable, and a leading ~ in workdir by the home directory; in
// mcp, Dial does the same when it connects.
// An unknown field is an error. Tools and ProviderImpl, being Go, are for
// the caller to add to the returned Config.
func LoadConfig(path string) (Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("agentkit: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var f fileConfig
	if err := dec.Decode(&f); err != nil {
		return Config{}, fmt.Errorf("agentkit: %s: %w", path, err)
	}
	if dec.More() {
		return Config{}, fmt.Errorf("agentkit: %s: more than one JSON value", path)
	}
	if f.Provider == "" {
		return Config{}, fmt.Errorf("agentkit: %s: no provider", path)
	}
	return Config{
		Provider:       f.Provider,
		Model:          f.Model,
		BaseURL:        expandEnv(f.BaseURL),
		APIKey:         expandEnv(f.APIKey),
		MaxSteps:       f.MaxSteps,
		MaxToolResult:  f.MaxToolResult,
		Temperature:    f.Temperature,
		MaxTokens:      f.MaxTokens,
		PromptCache:    f.PromptCache,
		ResponseSchema: f.ResponseSchema,
		MCP:            f.MCP,
		Builtins:       f.Tools,
		Workdir:        expandHome(expandEnv(f.Workdir)),
		System:         f.System,
		ExtraBody:      f.ExtraBody,
		Timeout:        time.Duration(f.Timeout) * time.Second,
	}, nil
}

// expandHome replaces a leading ~ by the home directory.
func expandHome(s string) string {
	if s != "~" && !strings.HasPrefix(s, "~/") {
		return s
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return s
	}
	return home + s[1:]
}

// expandEnv replaces every ${VAR} in s by the environment variable VAR,
// empty when unset.
func expandEnv(s string) string {
	return placeholder.ReplaceAllStringFunc(s, func(m string) string {
		return os.Getenv(m[2 : len(m)-1])
	})
}
