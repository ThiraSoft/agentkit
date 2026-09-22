package agentkit_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/ThiraSoft/agentkit"
	"github.com/ThiraSoft/agentkit/llm"
)

// scripted stands in for a model in these examples: it asks for the clock
// tool, then answers with what the tool returned.
type scripted struct{}

func (scripted) Name() string      { return "scripted" }
func (scripted) ModelName() string { return "scripted" }

func (s scripted) Chat(ctx context.Context, msgs []llm.Message, tools []llm.Tool) (*llm.Message, error) {
	return s.Stream(ctx, msgs, tools, func(string) error { return nil })
}

func (scripted) Stream(_ context.Context, msgs []llm.Message, _ []llm.Tool, onChunk func(string) error) (*llm.Message, error) {
	last := msgs[len(msgs)-1]
	if last.Role != "tool" {
		return &llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{{
			ID:       "1",
			Type:     "function",
			Function: llm.FunctionCall{Name: "clock", Arguments: json.RawMessage(`{}`)},
		}}}, nil
	}
	text := "It is " + last.Content + "."
	if err := onChunk(text); err != nil {
		return nil, err
	}
	return &llm.Message{Role: "assistant", Content: text}, nil
}

var clock = agentkit.Tool{
	Name:        "clock",
	Description: "Tells the time.",
	Parameters:  llm.ToolParams{Type: "object", Properties: llm.ToolProperties{}},
	Run: func(ctx context.Context, args json.RawMessage) (string, error) {
		return "12:00", nil
	},
}

func clockAgent() *agentkit.Agent {
	agent, err := agentkit.New(context.Background(), agentkit.Config{
		ProviderImpl: scripted{},
		Tools:        []agentkit.Tool{clock},
	})
	if err != nil {
		log.Fatal(err)
	}
	return agent
}

// An agent with one Go tool. A real program sets Provider and Model (for
// instance "gemini" and "gemini-2.5-flash-lite") instead of ProviderImpl.
func Example() {
	ctx := context.Background()
	agent, err := agentkit.New(ctx, agentkit.Config{
		ProviderImpl: scripted{},
		Tools:        []agentkit.Tool{clock},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer agent.Close()

	conv := agent.NewConversation("Answer in one sentence.")
	turn, err := conv.Send(ctx, "What time is it?", agentkit.Hooks{
		OnText: func(chunk string) { fmt.Println("text:", chunk) },
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("steps:", turn.Steps)
	fmt.Println("reply:", turn.Text)
	// Output:
	// text: It is 12:00.
	// steps: 2
	// reply: It is 12:00.
}

func ExampleHooks() {
	agent := clockAgent()
	defer agent.Close()

	_, err := agent.NewConversation("").Send(context.Background(), "What time is it?", agentkit.Hooks{
		OnToolCall:   func(c agentkit.ToolCall) { fmt.Println("call:", c.Name, string(c.Arguments)) },
		Approve:      func(c agentkit.ToolCall) bool { return c.Name == "clock" },
		OnToolResult: func(r agentkit.ToolResult) { fmt.Println("result:", r.Content) },
		OnFinish:     func() { fmt.Println("finished") },
	})
	if err != nil {
		log.Fatal(err)
	}
	// Output:
	// call: clock {}
	// result: 12:00
	// finished
}

func ExampleAgent_Tools() {
	agent := clockAgent()
	defer agent.Close()
	fmt.Println(agent.Tools())
	// Output: [clock]
}

func ExampleConversation_Messages() {
	agent := clockAgent()
	defer agent.Close()

	conv := agent.NewConversation("Answer in one sentence.")
	if _, err := conv.Send(context.Background(), "What time is it?", agentkit.Hooks{}); err != nil {
		log.Fatal(err)
	}
	var roles []string
	for _, m := range conv.Messages() {
		roles = append(roles, m.Role)
	}
	fmt.Println(strings.Join(roles, " "))
	// Output: system user assistant tool assistant
}
