// thinking.go: Config.Thinking, how each provider turns a model's reasoning
// on or off. The model's name says which knob it takes; a model with no knob,
// or none that can do what is asked, gets nothing and keeps its default.

package llm

import (
	"regexp"
	"strconv"
	"strings"
)

// enableThinking is llama.cpp's knob, chat_template_kwargs.enable_thinking,
// which the model's template reads; golem, vLLM and the MLX servers read it
// too. It is added to fields, keeping the other chat_template_kwargs.
func enableThinking(fields map[string]any, on bool) {
	kwargs := map[string]any{}
	if m, ok := fields["chat_template_kwargs"].(map[string]any); ok {
		for k, v := range m {
			kwargs[k] = v
		}
	}
	kwargs["enable_thinking"] = on
	fields["chat_template_kwargs"] = kwargs
}

// openAIEffort is the reasoning_effort that turns an OpenAI reasoning model's
// thinking on or off, "" for a model that takes none. GPT-5.1 and after go
// down to "none", GPT-5 to "minimal", the o series to "low".
func openAIEffort(model string, on bool) string {
	m := strings.ToLower(model)
	if i := strings.LastIndex(m, "/"); i >= 0 {
		m = m[i+1:]
	}
	if on {
		if isOpenAIReasoning(m) {
			return "high"
		}
		return ""
	}
	switch {
	case !isOpenAIReasoning(m):
		return ""
	case strings.HasPrefix(m, "gpt-5.") || strings.HasPrefix(m, "gpt-6") || strings.HasPrefix(m, "gpt-7"):
		return "none"
	case strings.HasPrefix(m, "gpt-5"):
		return "minimal"
	default:
		return "low"
	}
}

func isOpenAIReasoning(m string) bool {
	if strings.Contains(m, "chat") {
		return false // gpt-5-chat-latest and the like refuse reasoning_effort
	}
	for _, p := range []string{"o1", "o3", "o4", "gpt-5", "gpt-6", "gpt-7"} {
		if strings.HasPrefix(m, p) {
			return true
		}
	}
	return false
}

// ollamaThink is the think field of an Ollama request: a bool, but a level
// for gpt-oss, which takes no false.
func ollamaThink(model string, on bool) any {
	if strings.HasPrefix(strings.ToLower(model), "gpt-oss") {
		if on {
			return "high"
		}
		return "low"
	}
	return on
}

// geminiThinking is the thinkingConfig of a Gemini request, nil when the
// model takes none. Gemini 3 takes a level and cannot stop thinking: off is
// its lowest. Gemini 2.5 takes a budget, which 2.5 Pro cannot set to 0.
func geminiThinking(model string, on bool) map[string]any {
	m := strings.ToLower(model)
	major, minor, ok := geminiVersion(m)
	switch {
	case !ok:
		return nil
	case major >= 3:
		level := "high"
		if !on {
			level = "low"
			if strings.Contains(m, "flash") {
				level = "minimal"
			}
		}
		return map[string]any{"thinkingLevel": level, "includeThoughts": on}
	case major == 2 && minor >= 5:
		if on {
			return map[string]any{"thinkingBudget": -1, "includeThoughts": true}
		}
		if strings.Contains(m, "pro") {
			return nil
		}
		return map[string]any{"thinkingBudget": 0}
	}
	return nil
}

var geminiName = regexp.MustCompile(`gemini-(\d+)(?:\.(\d+))?`)

func geminiVersion(m string) (major, minor int, ok bool) {
	g := geminiName.FindStringSubmatch(m)
	if g == nil {
		return 0, 0, false
	}
	major, _ = strconv.Atoi(g[1])
	minor, _ = strconv.Atoi(g[2])
	return major, minor, true
}

// claudeThinking says how a Claude model is asked to think, or not:
// the thinking field, an output_config.effort, or nothing to send. Which
// one depends on the model:
//   - Fable, Mythos and Opus 5 and after always think; off is effort "low".
//   - Sonnet 5.5 and after turn off with "between_tools".
//   - Opus 4.6 to 4.8, Sonnet 4.6 and Sonnet 5 take "disabled".
//   - All of these think adaptively when on, the summary of it returned.
//   - Claude 3.7 to 4.5 think with a budget, below maxTokens, and not at all
//     by default.
//
// Older models have no thinking.
func claudeThinking(model string, on bool, maxTokens int) (thinking map[string]any, effort string) {
	family, major, minor, ok := claudeVersion(model)
	if !ok {
		return nil, ""
	}
	v := major*10 + minor
	adaptive := family == "fable" || family == "mythos" || v >= 50 ||
		(v >= 46 && (family == "opus" || family == "sonnet"))
	switch {
	case adaptive && on:
		return map[string]any{"type": "adaptive", "display": "summarized"}, ""
	case family == "fable" || family == "mythos" || (family == "opus" && v >= 50):
		return nil, "low"
	case family == "sonnet" && v >= 55:
		return map[string]any{"type": "between_tools"}, ""
	case adaptive:
		return map[string]any{"type": "disabled"}, ""
	case v >= 37 && on:
		budget := maxTokens / 2
		if budget > 16000 {
			budget = 16000
		}
		if budget < 1024 {
			return nil, "" // too few tokens to think in
		}
		return map[string]any{"type": "enabled", "budget_tokens": budget}, ""
	}
	return nil, ""
}

var (
	claudeName    = regexp.MustCompile(`claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?(?:$|[^0-9])`)
	claudeOldName = regexp.MustCompile(`claude-(\d)-(\d)-(opus|sonnet|haiku)`)
)

// claudeVersion reads the family and version of a Claude model's name:
// claude-opus-4-8, claude-sonnet-4-5-20250929, claude-3-7-sonnet-latest...
func claudeVersion(model string) (family string, major, minor int, ok bool) {
	m := strings.ToLower(model)
	if g := claudeName.FindStringSubmatch(m); g != nil {
		major, _ = strconv.Atoi(g[2])
		minor, _ = strconv.Atoi(g[3])
		return g[1], major, minor, true
	}
	if g := claudeOldName.FindStringSubmatch(m); g != nil {
		major, _ = strconv.Atoi(g[1])
		minor, _ = strconv.Atoi(g[2])
		return g[3], major, minor, true
	}
	return "", 0, 0, false
}
