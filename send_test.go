package agentkit

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/agentkit/llm"
)

func newTestAgent(t *testing.T, f *fakeProvider, cfg Config) *Agent {
	t.Helper()
	cfg.ProviderImpl = f
	a, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { a.Close() })
	return a
}

func roles(msgs []llm.Message) string {
	var r []string
	for _, m := range msgs {
		r = append(r, m.Role)
	}
	return strings.Join(r, ",")
}

func TestSendPlain(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{{chunks: []string{"Hel", "lo"}}}}
	conv := newTestAgent(t, f, Config{}).NewConversation("system")
	var got strings.Builder
	turn, err := conv.Send(context.Background(), "Hi", Hooks{OnText: func(c string) { got.WriteString(c) }})
	if err != nil {
		t.Fatal(err)
	}
	if got.String() != "Hello" || turn.Text != "Hello" || turn.Steps != 1 || turn.Interrupted {
		t.Fatalf("text %q, turn %+v", got.String(), turn)
	}
	msgs := conv.Messages()
	if roles(msgs) != "system,user,assistant" || msgs[0].Content != "system" || msgs[1].Content != "Hi" {
		t.Fatalf("history %+v", msgs)
	}
}

func TestToolsRunInParallelAndInOrder(t *testing.T) {
	fastDone := make(chan struct{})
	slow := echoTool("slow", func(ctx context.Context, _ json.RawMessage) (string, error) {
		select {
		case <-fastDone: // only finishes after the other: proves parallelism
			return "slow done", nil
		case <-time.After(2 * time.Second):
			return "", errors.New("tools do not run in parallel")
		}
	})
	fast := echoTool("fast", func(context.Context, json.RawMessage) (string, error) {
		close(fastDone)
		return "fast done", nil
	})
	f := &fakeProvider{replies: []fakeReply{
		{chunks: []string{"Looking."}, calls: []llm.ToolCall{call("1", "slow", `{}`), call("2", "fast", `{}`)}},
		{chunks: []string{"Done."}},
	}}
	conv := newTestAgent(t, f, Config{Tools: []Tool{slow, fast}}).NewConversation("s")
	var events []string
	turn, err := conv.Send(context.Background(), "Go", Hooks{
		OnStepEnd:    func() { events = append(events, "step end") },
		OnToolCall:   func(c ToolCall) { events = append(events, "call "+c.Name) },
		OnToolResult: func(r ToolResult) { events = append(events, "result "+r.Name) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := "step end,call slow,call fast,result slow,result fast"; strings.Join(events, ",") != want {
		t.Fatalf("events %v, want %s", events, want)
	}
	msgs := conv.Messages()
	if roles(msgs) != "system,user,assistant,tool,tool,assistant" {
		t.Fatalf("history %s", roles(msgs))
	}
	if msgs[3].ToolID != "1" || msgs[3].Content != "slow done" || msgs[4].ToolID != "2" || msgs[4].Content != "fast done" {
		t.Fatalf("results %+v %+v", msgs[3], msgs[4])
	}
	if turn.Text != "Looking.\n\nDone." || turn.Steps != 2 {
		t.Fatalf("turn %+v", turn)
	}
}

func TestToolResults(t *testing.T) {
	long := strings.Repeat("é", 20) // 40 bytes
	tools := []Tool{
		echoTool("long", func(context.Context, json.RawMessage) (string, error) { return long, nil }),
		echoTool("crash", func(context.Context, json.RawMessage) (string, error) { return "", errors.New("boom") }),
		echoTool("forbidden", func(context.Context, json.RawMessage) (string, error) {
			t.Error("a refused tool ran")
			return "", nil
		}),
	}
	f := &fakeProvider{replies: []fakeReply{
		{calls: []llm.ToolCall{call("1", "long", `{}`), call("2", "crash", `{}`), call("3", "nope", `{}`), call("4", "forbidden", `{}`)}},
		{chunks: []string{"ok"}},
	}}
	conv := newTestAgent(t, f, Config{Tools: tools, MaxToolResult: 11}).NewConversation("s")
	var results []ToolResult
	_, err := conv.Send(context.Background(), "x", Hooks{
		Approve:      func(c ToolCall) bool { return c.Name != "forbidden" },
		OnToolResult: func(r ToolResult) { results = append(results, r) },
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		strings.Repeat("é", 5) + "\n[truncated, 40 bytes]", // 11 bytes cut on a UTF-8 boundary: 10
		"error: boom",
		"error: unknown tool nope",
		"declined by user",
	}
	for i, w := range want {
		if results[i].Content != w {
			t.Errorf("result %d %q, want %q", i, results[i].Content, w)
		}
	}
	if results[1].Err != "boom" || results[0].Err != "" || results[3].Err != "declined by user" {
		t.Fatalf("errs %q %q %q", results[0].Err, results[1].Err, results[3].Err)
	}
}

func TestMaxSteps(t *testing.T) {
	loop := fakeReply{calls: []llm.ToolCall{call("1", "a", `{}`)}}
	f := &fakeProvider{replies: []fakeReply{loop, loop, loop}}
	conv := newTestAgent(t, f, Config{Tools: []Tool{echoTool("a", ok)}, MaxSteps: 2}).NewConversation("s")
	turn, err := conv.Send(context.Background(), "x", Hooks{})
	if !errors.Is(err, ErrMaxSteps) || turn.Steps != 2 {
		t.Fatalf("turn %+v, err %v", turn, err)
	}
	if roles(conv.Messages()) != "system,user,assistant,tool,assistant,tool,assistant" {
		t.Fatalf("history %s", roles(conv.Messages()))
	}
	if last := conv.Messages()[len(conv.Messages())-1]; last.Content != "…" {
		t.Fatalf("last message %+v, want placeholder", last)
	}
}

func TestPrepareIsEphemeral(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{{chunks: []string{"ok"}}}}
	conv := newTestAgent(t, f, Config{}).NewConversation("s")
	_, err := conv.Send(context.Background(), "question", Hooks{Prepare: func(msgs []llm.Message) []llm.Message {
		msgs[len(msgs)-1].Content = "(memory) " + msgs[len(msgs)-1].Content
		return msgs
	}})
	if err != nil {
		t.Fatal(err)
	}
	if got := f.request(0)[1].Content; got != "(memory) question" {
		t.Fatalf("the model was asked %q", got)
	}
	if got := conv.Messages()[1].Content; got != "question" {
		t.Fatalf("history keeps %q", got)
	}
}

func TestProviderErrorKeepsThePartialText(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{{
		chunks: []string{"Mine"},
		calls:  []llm.ToolCall{call("1", "incomplete", `{}`)},
		err:    errors.New("network"),
	}}}
	conv := newTestAgent(t, f, Config{}).NewConversation("s")
	turn, err := conv.Send(context.Background(), "x", Hooks{})
	if err == nil || turn.Interrupted || turn.Text != "Mine" {
		t.Fatalf("turn %+v, err %v", turn, err)
	}
	msgs := conv.Messages()
	if roles(msgs) != "system,user,assistant" || len(msgs[2].ToolCalls) != 0 {
		t.Fatalf("history %+v", msgs)
	}
}

func TestSetSystemAndReset(t *testing.T) {
	f := &fakeProvider{}
	conv := newTestAgent(t, f, Config{}).NewConversation("before", llm.Message{Role: "user", Content: "Hello"}, llm.Message{Role: "assistant", Content: "Hi"})
	conv.SetSystem("after")
	if msgs := conv.Messages(); len(msgs) != 3 || msgs[0].Content != "after" {
		t.Fatalf("history %+v", msgs)
	}
	conv.Reset()
	if msgs := conv.Messages(); len(msgs) != 1 || msgs[0].Content != "after" {
		t.Fatalf("after reset %+v", msgs)
	}
}

func TestSendRecoverPanicAllowsSubsequentSend(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{
		{chunks: []string{"One"}},
		{chunks: []string{"Two"}},
	}}
	conv := newTestAgent(t, f, Config{}).NewConversation("s")

	// First Send whose OnText panics
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic from OnText")
			}
		}()
		_, _ = conv.Send(context.Background(), "first", Hooks{
			OnText: func(string) {
				panic("boom in hook")
			},
		})
	}()

	// The second Send must succeed without looping in stopRunning
	done := make(chan Turn, 1)
	errCh := make(chan error, 1)
	go func() {
		turn, err := conv.Send(context.Background(), "second", Hooks{})
		if err != nil {
			errCh <- err
			return
		}
		done <- turn
	}()

	select {
	case <-time.After(2 * time.Second):
		t.Fatal("second Send blocked (infinite loop in stopRunning)")
	case err := <-errCh:
		t.Fatalf("second Send failed: %v", err)
	case turn := <-done:
		if turn.Text != "Two" {
			t.Fatalf("turn text %q, want 'Two'", turn.Text)
		}
	}
}

func TestStepEndCancelSkipsApprove(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{
		{calls: []llm.ToolCall{call("1", "tool", `{}`)}},
	}}
	conv := newTestAgent(t, f, Config{Tools: []Tool{echoTool("tool", ok)}}).NewConversation("s")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var approveCalled bool
	var gotResults []ToolResult
	turn, err := conv.Send(ctx, "x", Hooks{
		OnStepEnd: func() {
			cancel()
		},
		Approve: func(ToolCall) bool {
			approveCalled = true
			return true
		},
		OnToolResult: func(r ToolResult) {
			gotResults = append(gotResults, r)
		},
	})
	if !turn.Interrupted || !errors.Is(err, context.Canceled) {
		t.Fatalf("turn %+v, err %v", turn, err)
	}
	if approveCalled {
		t.Fatal("Approve was called even though context was already canceled in OnStepEnd")
	}
	if len(gotResults) != 1 || gotResults[0].Content != "interrupted" || gotResults[0].Err != context.Canceled.Error() {
		t.Fatalf("results %+v", gotResults)
	}
}

func TestProviderErrorWithoutTextLeavesPlaceholder(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{{err: errors.New("immediate network error")}}}
	conv := newTestAgent(t, f, Config{}).NewConversation("s")
	turn, err := conv.Send(context.Background(), "hello", Hooks{})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if turn.Text != "" {
		t.Fatalf("turn text %q, want empty string", turn.Text)
	}
	msgs := conv.Messages()
	if roles(msgs) != "system,user,assistant" || msgs[2].Content != "…" {
		t.Fatalf("history %+v, want assistant placeholder", msgs)
	}
}

func TestEmptyResponseLeavesPlaceholder(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{{}}}
	conv := newTestAgent(t, f, Config{}).NewConversation("s")
	turn, err := conv.Send(context.Background(), "hello", Hooks{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if turn.Text != "" {
		t.Fatalf("turn text %q, want empty string", turn.Text)
	}
	msgs := conv.Messages()
	if roles(msgs) != "system,user,assistant" || msgs[2].Content != "…" {
		t.Fatalf("history %+v, want assistant placeholder", msgs)
	}
}

func TestEmptyResponseAfterToolLeavesPlaceholder(t *testing.T) {
	tools := []Tool{echoTool("a", ok)}
	f := &fakeProvider{replies: []fakeReply{
		{calls: []llm.ToolCall{call("1", "a", `{}`)}},
		{},
	}}
	conv := newTestAgent(t, f, Config{Tools: tools}).NewConversation("s")
	turn, err := conv.Send(context.Background(), "hello", Hooks{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if turn.Text != "" {
		t.Fatalf("turn text %q, want empty string", turn.Text)
	}
	msgs := conv.Messages()
	if roles(msgs) != "system,user,assistant,tool,assistant" || msgs[len(msgs)-1].Content != "…" {
		t.Fatalf("history %+v, want assistant placeholder", msgs)
	}
}
