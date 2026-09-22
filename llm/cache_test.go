package llm

import "testing"

func TestAnthropicPromptCache(t *testing.T) {
	b := sendWith(t, Config{Provider: ProviderAnthropic, PromptCache: true}, "be brief")
	cc, _ := b["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" {
		t.Fatalf("top-level cache_control %v", b["cache_control"])
	}
	blocks, _ := b["system"].([]any)
	if len(blocks) != 1 {
		t.Fatalf("system %v", b["system"])
	}
	block := blocks[0].(map[string]any)
	if block["type"] != "text" || block["text"] != "be brief\n" || block["cache_control"].(map[string]any)["type"] != "ephemeral" {
		t.Fatalf("system block %v", block)
	}

	b = sendWith(t, Config{Provider: ProviderAnthropic}, "be brief")
	if _, has := b["cache_control"]; has {
		t.Fatalf("cache_control without PromptCache: %v", b)
	}
	if b["system"] != "be brief\n" {
		t.Fatalf("system without PromptCache %v", b["system"])
	}

	b = sendWith(t, Config{Provider: ProviderAnthropic, PromptCache: true}, "")
	if _, has := b["system"]; has {
		t.Fatalf("system sent without a system prompt: %v", b["system"])
	}
	if _, has := b["cache_control"]; !has {
		t.Fatal("no cache_control without a system prompt")
	}
}
