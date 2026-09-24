package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/ThiraSoft/agentkit"
	"github.com/ThiraSoft/agentkit/llm"
)

const help = "/reset clears the history, /usage shows the tokens spent, /tools lists the tools, /quit leaves"

// repl reads lines and sends each to conv, the answer streamed to out and
// the tool calls and the rest to errOut. turnCtx gives the context of each
// turn, so that main can cut one on Ctrl-C.
type repl struct {
	agent   *agentkit.Agent
	conv    *agentkit.Conversation
	out     io.Writer
	errOut  io.Writer
	usage   llm.Usage
	turnCtx func() (context.Context, context.CancelFunc)
	// save, when set, keeps the history after each turn and each /reset.
	save func() error
	// final prints on out only the last message of the turn, the answer,
	// and what came before it nowhere: what -p does. verbose sends that to
	// errOut instead.
	final, verbose bool
}

// run reads in until it ends or a line says /quit.
func (r *repl) run(in io.Reader) error {
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 0, 64<<10), 1<<20)
	for {
		fmt.Fprint(r.out, "> ")
		if !sc.Scan() {
			fmt.Fprintln(r.out)
			return sc.Err()
		}
		line := strings.TrimSpace(sc.Text())
		switch line {
		case "":
		case "/quit", "/exit":
			return nil
		case "/help":
			fmt.Fprintln(r.errOut, help)
		case "/reset":
			r.conv.Reset()
			r.usage = llm.Usage{}
			r.keep()
			fmt.Fprintln(r.errOut, "(history cleared)")
		case "/usage":
			u := r.usage
			fmt.Fprintf(r.errOut, "input %d, output %d, cache read %d, cache write %d\n",
				u.InputTokens, u.OutputTokens, u.CacheReadTokens, u.CacheWriteTokens)
		case "/tools":
			tools := r.agent.Tools()
			if len(tools) == 0 {
				fmt.Fprintln(r.errOut, "(no tools)")
			} else {
				fmt.Fprintln(r.errOut, strings.Join(tools, "\n"))
			}
		default:
			r.send(line)
		}
	}
}

// keep saves the history, if there is somewhere to.
func (r *repl) keep() bool {
	if r.save == nil {
		return true
	}
	if err := r.save(); err != nil {
		fmt.Fprintf(r.errOut, "session not saved: %v\n", err)
		return false
	}
	return true
}

// send runs one turn and tells how it ended, and whether it went to the end.
func (r *repl) send(text string) bool {
	ctx, cancel := r.turnCtx()
	defer cancel()
	stream, trace := r.out, r.errOut
	if r.final {
		stream, trace = io.Discard, io.Discard
		if r.verbose {
			stream, trace = r.errOut, r.errOut
		}
	}
	turn, err := r.conv.Send(ctx, text, agentkit.Hooks{
		OnText: func(chunk string) { fmt.Fprint(stream, chunk) },
		OnToolCall: func(c agentkit.ToolCall) {
			fmt.Fprintf(trace, "\n[%s %s]\n", c.Name, c.Arguments)
		},
		OnToolResult: func(res agentkit.ToolResult) {
			if res.Err != "" {
				fmt.Fprintf(trace, "[%s failed: %s]\n", res.Name, res.Err)
				return
			}
			fmt.Fprintf(trace, "[%s: %d bytes]\n", res.Name, len(res.Content))
		},
	})
	fmt.Fprintln(stream)
	if r.final && err == nil && !turn.Interrupted {
		msgs := r.conv.Messages()
		answer := strings.TrimSpace(msgs[len(msgs)-1].Content)
		// A model that stops without a word, its tokens spent on thinking
		// for instance, has not done the task: that is no success.
		if answer == "" || answer == "…" {
			err = errors.New("the model ended its turn without an answer")
		} else {
			fmt.Fprintln(r.out, answer)
		}
	}
	r.usage = r.usage.Add(turn.Usage)
	if r.verbose {
		u := r.usage
		fmt.Fprintf(r.errOut, "(%d steps, input %d, output %d)\n", turn.Steps, u.InputTokens, u.OutputTokens)
	}
	saved := r.keep()
	switch {
	case turn.Interrupted:
		fmt.Fprintln(r.errOut, "(interrupted)")
		return false
	case err != nil:
		fmt.Fprintf(r.errOut, "error: %v\n", err)
		return false
	}
	return saved
}
