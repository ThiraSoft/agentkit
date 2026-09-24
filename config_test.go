package agentkit

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeConfig(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "agent.json")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadConfig(t *testing.T) {
	t.Setenv("AK_TEST_KEY", "secret")
	t.Setenv("AK_TEST_HOST", "localhost:8080")
	path := writeConfig(t, `{
		"provider": "openai-compat",
		"model": "local",
		"baseURL": "http://${AK_TEST_HOST}/v1",
		"apiKey": "${AK_TEST_KEY}",
		"maxSteps": 7,
		"maxToolResult": 1024,
		"temperature": 0.5,
		"maxTokens": 256,
		"promptCache": true,
		"responseSchema": {"type": "object"},
		"mcp": [{"name": "files", "transport": "stdio", "command": "${HOME}/bin/files-mcp"}]
	}`)
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Provider != "openai-compat" || cfg.Model != "local" || cfg.BaseURL != "http://localhost:8080/v1" || cfg.APIKey != "secret" {
		t.Fatalf("provider fields %+v", cfg)
	}
	if cfg.MaxSteps != 7 || cfg.MaxToolResult != 1024 || cfg.Temperature == nil || *cfg.Temperature != 0.5 || cfg.MaxTokens != 256 || !cfg.PromptCache {
		t.Fatalf("options %+v", cfg)
	}
	if string(cfg.ResponseSchema) != `{"type": "object"}` {
		t.Fatalf("ResponseSchema %s", cfg.ResponseSchema)
	}
	// ${VAR} in mcp is left to mcp.Dial, which expands it when it connects.
	if len(cfg.MCP) != 1 || cfg.MCP[0].Command != "${HOME}/bin/files-mcp" {
		t.Fatalf("mcp %+v", cfg.MCP)
	}
}

func TestLoadConfigRefuses(t *testing.T) {
	cases := map[string]string{
		"unknown field": `{"provider": "gemini", "modle": "x"}`,
		"no provider":   `{"model": "x"}`,
		"not JSON":      `{`,
		"two objects":   `{"provider": "gemini"} {"provider": "openai"}`,
	}
	for name, content := range cases {
		if _, err := LoadConfig(writeConfig(t, content)); err == nil || !strings.HasPrefix(err.Error(), "agentkit: ") {
			t.Errorf("%s: err %v", name, err)
		}
	}
	if _, err := LoadConfig(filepath.Join(t.TempDir(), "absent.json")); err == nil {
		t.Error("a missing file was read")
	}
}

func TestLoadConfigTools(t *testing.T) {
	t.Setenv("HOME", "/home/someone")
	t.Setenv("AK_TEST_REPO", "src/repo")
	cfg, err := LoadConfig(writeConfig(t, `{
		"provider": "gemini",
		"tools": ["read_file", "bash"],
		"workdir": "~/${AK_TEST_REPO}",
		"system": "Be brief.",
		"extraBody": {"top_p": 0.8},
		"timeout": 1800
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(cfg.Builtins, ",") != "read_file,bash" || cfg.Workdir != "/home/someone/src/repo" || cfg.System != "Be brief." || cfg.ExtraBody["top_p"] != 0.8 || cfg.Timeout != 30*time.Minute {
		t.Fatalf("%+v", cfg)
	}
	cfg, err = LoadConfig(writeConfig(t, `{"provider": "gemini"}`))
	if err != nil || cfg.Builtins != nil || cfg.Workdir != "" {
		t.Fatalf("%+v %v", cfg, err)
	}
}

// The example in the repository stays a config LoadConfig reads, with tools
// New knows.
func TestExampleImplementer(t *testing.T) {
	cfg, err := LoadConfig("examples/implementer.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Builtins) == 0 || cfg.System == "" {
		t.Fatalf("%+v", cfg)
	}
	if _, err := BuiltinTools(t.TempDir(), cfg.Builtins); err != nil {
		t.Fatal(err)
	}
}
