package main

import (
	"bufio"
	"context"
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

// send runs one turn and tells how it ended.
func (r *repl) send(text string) {
	ctx, cancel := r.turnCtx()
	defer cancel()
	turn, err := r.conv.Send(ctx, text, agentkit.Hooks{
		OnText: func(chunk string) { fmt.Fprint(r.out, chunk) },
		OnToolCall: func(c agentkit.ToolCall) {
			fmt.Fprintf(r.errOut, "\n[%s %s]\n", c.Name, c.Arguments)
		},
		OnToolResult: func(res agentkit.ToolResult) {
			if res.Err != "" {
				fmt.Fprintf(r.errOut, "[%s failed: %s]\n", res.Name, res.Err)
				return
			}
			fmt.Fprintf(r.errOut, "[%s: %d bytes]\n", res.Name, len(res.Content))
		},
	})
	fmt.Fprintln(r.out)
	r.usage = r.usage.Add(turn.Usage)
	switch {
	case turn.Interrupted:
		fmt.Fprintln(r.errOut, "(interrupted)")
	case err != nil:
		fmt.Fprintf(r.errOut, "error: %v\n", err)
	}
}
