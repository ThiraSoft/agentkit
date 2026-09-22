package llm

import (
	"encoding/json"
	"testing"
)

func TestResponseSchemaReachesEveryProvider(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`)
	hasAnswer := func(s any) bool {
		m, _ := s.(map[string]any)
		props, _ := m["properties"].(map[string]any)
		_, ok := props["answer"]
		return ok
	}
	compat := func(b map[string]any) bool {
		rf, _ := b["response_format"].(map[string]any)
		js, _ := rf["json_schema"].(map[string]any)
		return rf["type"] == "json_schema" && js["name"] == "response" && hasAnswer(js["schema"])
	}
	cases := []struct {
		provider ProviderType
		ok       func(map[string]any) bool
	}{
		{ProviderOpenAI, compat},
		{ProviderOpenAICompat, compat},
		{ProviderMistral, compat},
		{ProviderLlamaCpp, compat},
		{ProviderAnthropic, func(b map[string]any) bool {
			oc, _ := b["output_config"].(map[string]any)
			f, _ := oc["format"].(map[string]any)
			return f["type"] == "json_schema" && hasAnswer(f["schema"])
		}},
		{ProviderGemini, func(b map[string]any) bool {
			gc, _ := b["generationConfig"].(map[string]any)
			return gc["responseMimeType"] == "application/json" && hasAnswer(gc["responseJsonSchema"])
		}},
		{ProviderOllama, func(b map[string]any) bool { return hasAnswer(b["format"]) }},
	}
	for _, c := range cases {
		if b := send(t, Config{Provider: c.provider, ResponseSchema: schema}); !c.ok(b) {
			t.Errorf("%s: body %v", c.provider, b)
		}
		b := send(t, Config{Provider: c.provider})
		for _, k := range []string{"response_format", "output_config", "format"} {
			if _, has := b[k]; has {
				t.Errorf("%s sent %s without ResponseSchema", c.provider, k)
			}
		}
		if gc, ok := b["generationConfig"].(map[string]any); ok {
			for _, k := range []string{"responseMimeType", "responseJsonSchema"} {
				if _, has := gc[k]; has {
					t.Errorf("%s sent generationConfig.%s without ResponseSchema", c.provider, k)
				}
			}
		}
	}
}
