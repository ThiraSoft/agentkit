package agentkit

import "github.com/ThiraSoft/agentkit/llm"

// KeepTurns returns a Prepare hook that sends the model the system prompt
// and the last n turns of the history, a turn starting at a user message.
// The history itself stays whole. n below 1 counts as 1: the turn under
// way is always sent. Cutting at a user message never separates a tool
// call from its result.
//
// To combine it with a Prepare of your own, call it from there:
//
//	keep := agentkit.KeepTurns(10)
//	hooks.Prepare = func(msgs []llm.Message) []llm.Message { return mine(keep(msgs)) }
func KeepTurns(n int) func([]llm.Message) []llm.Message {
	if n < 1 {
		n = 1
	}
	return func(msgs []llm.Message) []llm.Message {
		starts := turnStarts(msgs)
		if len(starts) <= n {
			return msgs
		}
		return cut(msgs, starts[len(starts)-n])
	}
}

// KeepTokens returns a Prepare hook that drops the oldest turns until
// what is sent fits in budget tokens, as EstimateTokens counts them. The
// system prompt and the turn under way are always sent, even over budget.
// The history itself stays whole.
func KeepTokens(budget int) func([]llm.Message) []llm.Message {
	return func(msgs []llm.Message) []llm.Message {
		starts := turnStarts(msgs)
		if len(starts) == 0 {
			return msgs
		}
		first := 0 // the first message that can be dropped
		if msgs[0].Role == "system" {
			first = 1
		}
		total := 0
		for _, m := range msgs {
			total += EstimateTokens(m)
		}
		from := first
		for i := 0; i < len(starts)-1 && total > budget; i++ {
			for _, m := range msgs[from:starts[i+1]] {
				total -= EstimateTokens(m)
			}
			from = starts[i+1]
		}
		if from == first {
			return msgs
		}
		return cut(msgs, from)
	}
}

// EstimateTokens is the rough count KeepTokens works with: a token per
// four bytes of text, tool name and tool arguments, 1000 per media, and 4
// for the message itself.
func EstimateTokens(m llm.Message) int {
	n := len(m.Content) + len(m.Name)
	for _, tc := range m.ToolCalls {
		n += len(tc.Function.Name) + len(tc.Function.Arguments)
	}
	return n/4 + 1000*len(m.Media) + 4
}

// turnStarts returns the indices of the user messages of msgs.
func turnStarts(msgs []llm.Message) []int {
	var starts []int
	for i, m := range msgs {
		if m.Role == "user" {
			starts = append(starts, i)
		}
	}
	return starts
}

// cut returns, in a new slice, the system prompt of msgs if it has one,
// followed by msgs[from:].
func cut(msgs []llm.Message, from int) []llm.Message {
	out := make([]llm.Message, 0, len(msgs)-from+1)
	if msgs[0].Role == "system" {
		out = append(out, msgs[0])
	}
	return append(out, msgs[from:]...)
}
