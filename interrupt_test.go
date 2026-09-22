package agentkit

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/ThiraSoft/agentkit/llm"
)

// A Send during another interrupts the first, and the new turn sees what
// OnInterrupt kept from it.
func TestSendCutsTheRunningOne(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{
		{chunks: []string{"One. ", "Two."}, block: make(chan struct{})}, // never finishes on its own
		{chunks: []string{"Seen."}},
	}}
	conv := newTestAgent(t, f, Config{}).NewConversation("s")
	streaming := make(chan struct{})
	var written []string
	type result struct {
		turn Turn
		err  error
	}
	first := make(chan result, 1)
	go func() {
		turn, err := conv.Send(context.Background(), "Hello", Hooks{
			OnText: func(c string) {
				if c == "Two." {
					close(streaming)
				}
			},
			OnInterrupt: func(w string) string {
				written = append(written, w)
				return "One."
			},
		})
		first <- result{turn, err}
	}()
	<-streaming
	turn, err := conv.Send(context.Background(), "Other", Hooks{})
	if err != nil || turn.Text != "Seen." {
		t.Fatalf("second turn %+v, %v", turn, err)
	}
	r := <-first
	if !r.turn.Interrupted || !errors.Is(r.err, context.Canceled) || r.turn.Text != "One." {
		t.Fatalf("first turn %+v, %v", r.turn, r.err)
	}
	if len(written) != 1 || written[0] != "One. Two." {
		t.Fatalf("OnInterrupt got %q", written)
	}
	msgs := conv.Messages()
	if roles(msgs) != "system,user,assistant,user,assistant" || msgs[2].Content != "One." || msgs[3].Content != "Other" {
		t.Fatalf("history %+v", msgs)
	}
	if asked := f.request(1); roles(asked) != "system,user,assistant,user" || asked[2].Content != "One." {
		t.Fatalf("the second turn asked %+v", asked)
	}
}

// Interrupted during a tool: the call gets result "interrupted", and kept
// text comes after, "…" if nothing else.
func TestInterruptDuringTool(t *testing.T) {
	started := make(chan struct{})
	wait := echoTool("wait", func(ctx context.Context, _ json.RawMessage) (string, error) {
		close(started)
		<-ctx.Done()
		return "", ctx.Err()
	})
	f := &fakeProvider{replies: []fakeReply{{calls: []llm.ToolCall{call("1", "wait", `{}`)}}}}
	conv := newTestAgent(t, f, Config{Tools: []Tool{wait}}).NewConversation("s")
	ctx, cancel := context.WithCancel(context.Background())
	go func() { <-started; cancel() }()
	turn, err := conv.Send(ctx, "x", Hooks{})
	if !turn.Interrupted || !errors.Is(err, context.Canceled) || turn.Text != "…" {
		t.Fatalf("turn %+v, %v", turn, err)
	}
	msgs := conv.Messages()
	if roles(msgs) != "system,user,assistant,tool,assistant" {
		t.Fatalf("history %s", roles(msgs))
	}
	if msgs[3].ToolID != "1" || msgs[3].Content != "interrupted" || msgs[4].Content != "…" {
		t.Fatalf("history %+v", msgs)
	}
}

// Interrupted during OnFinish, when the model already wrote everything:
// it is an interruption, and OnInterrupt decides what text is kept.
func TestCutDuringOnFinish(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{{chunks: []string{"One. Two. Three."}}}}
	conv := newTestAgent(t, f, Config{}).NewConversation("s")
	ctx, cancel := context.WithCancel(context.Background())
	var written string
	turn, err := conv.Send(ctx, "x", Hooks{
		OnFinish:    func() { cancel() }, // voice is cut before the end
		OnInterrupt: func(w string) string { written = w; return "One." },
	})
	if !turn.Interrupted || !errors.Is(err, context.Canceled) || written != "One. Two. Three." || turn.Text != "One." {
		t.Fatalf("turn %+v, err %v, written %q", turn, err, written)
	}
	if msgs := conv.Messages(); msgs[2].Content != "One." {
		t.Fatalf("history %+v", msgs)
	}
}

// Reset interrupts the running turn before clearing.
func TestResetCutsTheRunningTurn(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{{chunks: []string{"One."}, block: make(chan struct{})}}}
	conv := newTestAgent(t, f, Config{}).NewConversation("s")
	streaming := make(chan struct{})
	done := make(chan Turn, 1)
	go func() {
		turn, _ := conv.Send(context.Background(), "x", Hooks{OnText: func(string) { close(streaming) }})
		done <- turn
	}()
	<-streaming
	conv.Reset()
	if turn := <-done; !turn.Interrupted {
		t.Fatalf("turn %+v, want interrupted", turn)
	}
	if msgs := conv.Messages(); len(msgs) != 1 {
		t.Fatalf("history %+v, want the system prompt alone", msgs)
	}
}

// A turn whose OnFinish returned without interruption passes cut=false to end:
// a later cancellation must not mark the turn as interrupted.
func TestEndWithCutFalseIgnoresCanceledContext(t *testing.T) {
	conv := newTestAgent(t, &fakeProvider{}, Config{}).NewConversation("s")
	conv.add(llm.Message{Role: "assistant", Content: "Valid text"})
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // context canceled after OnFinish returned

	turn, err := conv.end(ctx, Hooks{}, 1, Turn{}, false, nil)
	if turn.Interrupted || err != nil || turn.Text != "Valid text" {
		t.Fatalf("turn %+v, err %v, want cut=false to preserve turn even if ctx is canceled", turn, err)
	}
}

// A second Send arriving while a tool runs interrupts the first turn:
// the first turn is Interrupted, its tool gets result "interrupted",
// kept text comes after, and the second turn sees all of it.
func TestSendCutsDuringTool(t *testing.T) {
	started := make(chan struct{})
	wait := echoTool("wait", func(ctx context.Context, _ json.RawMessage) (string, error) {
		close(started)
		<-ctx.Done()
		return "", ctx.Err()
	})
	f := &fakeProvider{replies: []fakeReply{
		{calls: []llm.ToolCall{call("1", "wait", `{}`)}},
		{chunks: []string{"Continued."}},
	}}
	conv := newTestAgent(t, f, Config{Tools: []Tool{wait}}).NewConversation("s")

	type result struct {
		turn Turn
		err  error
	}
	first := make(chan result, 1)
	go func() {
		turn, err := conv.Send(context.Background(), "First", Hooks{})
		first <- result{turn, err}
	}()

	<-started
	secondTurn, err := conv.Send(context.Background(), "Second", Hooks{})
	if err != nil || secondTurn.Text != "Continued." {
		t.Fatalf("second turn %+v, %v", secondTurn, err)
	}

	r := <-first
	if !r.turn.Interrupted || !errors.Is(r.err, context.Canceled) || r.turn.Text != "…" {
		t.Fatalf("first turn %+v, %v", r.turn, r.err)
	}

	msgs := conv.Messages()
	if roles(msgs) != "system,user,assistant,tool,assistant,user,assistant" {
		t.Fatalf("history roles %s", roles(msgs))
	}
	if msgs[3].Content != "interrupted" || msgs[4].Content != "…" || msgs[5].Content != "Second" || msgs[6].Content != "Continued." {
		t.Fatalf("history messages %+v", msgs)
	}

	asked := f.request(1)
	if roles(asked) != "system,user,assistant,tool,assistant,user" {
		t.Fatalf("second turn requested roles %s", roles(asked))
	}
	if asked[3].Content != "interrupted" || asked[4].Content != "…" {
		t.Fatalf("second turn requested %+v", asked)
	}
}

// An interrupted turn must always return ctx.Err(), even if the provider returned
// an unrelated error (e.g. network) after cancellation.
func TestSendCutOverridesProviderError(t *testing.T) {
	block := make(chan struct{})
	f := &fakeProvider{replies: []fakeReply{{
		chunks: []string{"start"},
		block:  block,
		err:    errors.New("network"),
	}}}
	conv := newTestAgent(t, f, Config{}).NewConversation("s")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	turn, err := conv.Send(ctx, "x", Hooks{
		OnText: func(string) {
			cancel()
			close(block)
		},
	})
	if !turn.Interrupted {
		t.Fatalf("turn %+v, want interrupted", turn)
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v, want context.Canceled", err)
	}
}
