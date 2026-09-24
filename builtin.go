package agentkit

// Tools that come with agentkit, for an agent that works on files: read,
// write, edit, list, search, and a shell. None is on unless the Config names
// it, since each one reaches the disk.

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

const (
	readLimit       = 2000 // lines read_file returns when not told
	grepLimit       = 200  // matches grep returns at most
	grepMaxFileSize = 2 << 20
	shellTimeout    = 2 * time.Minute
	shellMaxTimeout = 10 * time.Minute
)

// Builtins lists the names of the tools agentkit provides, in the order the
// model sees them.
func Builtins() []string {
	return []string{"read_file", "write_file", "edit_file", "list_dir", "grep", "bash"}
}

// BuiltinTools builds the tools named, working in dir. The file tools cannot
// reach outside dir, through .. or a link. bash runs sh -c in dir and can do
// anything the process can: name it only for an agent you would let type in
// your terminal.
func BuiltinTools(dir string, names []string) ([]Tool, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("agentkit: workdir: %w", err)
	}
	if info, err := os.Stat(abs); err != nil || !info.IsDir() {
		return nil, fmt.Errorf("agentkit: workdir %s is not a directory", abs)
	}
	w := workdir(abs)
	var tools []Tool
	for _, name := range names {
		var t Tool
		var err error
		switch name {
		case "read_file":
			t, err = NewTool("read_file", "Reads a text file of the working directory. Each line comes numbered, from 1. Give offset and limit to read part of a long file.", w.readFile)
		case "write_file":
			t, err = NewTool("write_file", "Writes a file of the working directory, whole, creating it and its directories if needed. Replaces what was there.", w.writeFile)
		case "edit_file":
			t, err = NewTool("edit_file", "Replaces old_string by new_string in a file of the working directory. old_string must appear exactly once, unless replace_all is true; copy it from read_file, without the line numbers.", w.editFile)
		case "list_dir":
			t, err = NewTool("list_dir", "Lists a directory of the working directory, directories with a trailing /. recursive lists all it holds, .git excepted.", w.listDir)
		case "grep":
			t, err = NewTool("grep", "Searches the files of the working directory for a regular expression (Go syntax), and gives path:line: text for each match.", w.grep)
		case "bash":
			t, err = NewTool("bash", "Runs a command with sh -c in the working directory and gives its output and exit code. Two minutes at most unless timeout_seconds says otherwise (ten at most).", w.bash)
		default:
			return nil, fmt.Errorf("agentkit: no built-in tool %q (there are %s)", name, strings.Join(Builtins(), ", "))
		}
		if err != nil {
			return nil, err
		}
		tools = append(tools, t)
	}
	return tools, nil
}

// workdir is the directory the tools work in, an absolute path.
type workdir string

// open gives the directory as an os.Root, which refuses a path that leaves it.
func (w workdir) open() (*os.Root, error) { return os.OpenRoot(string(w)) }

// rel makes p relative to the working directory, accepting an absolute path
// that lies inside it, since that is what a model often writes.
func (w workdir) rel(p string) string {
	if p == "" {
		return "."
	}
	if filepath.IsAbs(p) {
		if r, err := filepath.Rel(string(w), p); err == nil {
			return r
		}
	}
	return p
}

type readArgs struct {
	Path   string `json:"path" jsonschema:"the file, relative to the working directory"`
	Offset int    `json:"offset,omitempty" jsonschema:"the first line to read, from 1"`
	Limit  int    `json:"limit,omitempty" jsonschema:"how many lines, 2000 by default"`
}

func (w workdir) readFile(_ context.Context, a readArgs) (string, error) {
	root, err := w.open()
	if err != nil {
		return "", err
	}
	defer root.Close()
	data, err := root.ReadFile(w.rel(a.Path))
	if err != nil {
		return "", err
	}
	if bytes.IndexByte(data, 0) >= 0 {
		return "", fmt.Errorf("%s is not a text file", a.Path)
	}
	if len(data) == 0 {
		return "(empty file)", nil
	}
	lines := strings.SplitAfter(string(data), "\n")
	if lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	first := max(a.Offset, 1)
	limit := a.Limit
	if limit <= 0 {
		limit = readLimit
	}
	if first > len(lines) {
		return "", fmt.Errorf("%s has %d lines, none from %d", a.Path, len(lines), first)
	}
	last := min(first-1+limit, len(lines))
	var b strings.Builder
	for i := first - 1; i < last; i++ {
		fmt.Fprintf(&b, "%d\t%s", i+1, strings.TrimSuffix(lines[i], "\n"))
		b.WriteByte('\n')
	}
	if last < len(lines) {
		fmt.Fprintf(&b, "(%d lines more, from %d)\n", len(lines)-last, last+1)
	}
	return b.String(), nil
}

type writeArgs struct {
	Path    string `json:"path" jsonschema:"the file, relative to the working directory"`
	Content string `json:"content" jsonschema:"the whole content of the file"`
}

func (w workdir) writeFile(_ context.Context, a writeArgs) (string, error) {
	if a.Path == "" {
		return "", errors.New("no path")
	}
	root, err := w.open()
	if err != nil {
		return "", err
	}
	defer root.Close()
	p := w.rel(a.Path)
	if dir := filepath.Dir(p); dir != "." {
		if err := root.MkdirAll(dir, 0o755); err != nil {
			return "", err
		}
	}
	if err := root.WriteFile(p, []byte(a.Content), 0o644); err != nil {
		return "", err
	}
	return fmt.Sprintf("wrote %s, %d bytes", a.Path, len(a.Content)), nil
}

type editArgs struct {
	Path       string `json:"path" jsonschema:"the file, relative to the working directory"`
	OldString  string `json:"old_string" jsonschema:"the text to replace, exactly as in the file"`
	NewString  string `json:"new_string" jsonschema:"the text to put in its place"`
	ReplaceAll bool   `json:"replace_all,omitempty" jsonschema:"replace every occurrence, not just a single one"`
}

func (w workdir) editFile(_ context.Context, a editArgs) (string, error) {
	if a.OldString == "" {
		return "", errors.New("old_string is empty; write_file makes a new file")
	}
	if a.OldString == a.NewString {
		return "", errors.New("old_string and new_string are the same")
	}
	root, err := w.open()
	if err != nil {
		return "", err
	}
	defer root.Close()
	p := w.rel(a.Path)
	data, err := root.ReadFile(p)
	if err != nil {
		return "", err
	}
	text := string(data)
	n := strings.Count(text, a.OldString)
	switch {
	case n == 0:
		return "", fmt.Errorf("old_string is not in %s", a.Path)
	case n > 1 && !a.ReplaceAll:
		return "", fmt.Errorf("old_string is %d times in %s; give more of the text around it, or replace_all", n, a.Path)
	}
	text = strings.ReplaceAll(text, a.OldString, a.NewString)
	info, err := root.Stat(p)
	if err != nil {
		return "", err
	}
	if err := root.WriteFile(p, []byte(text), info.Mode().Perm()); err != nil {
		return "", err
	}
	return fmt.Sprintf("replaced %d occurrence(s) in %s", n, a.Path), nil
}

type listArgs struct {
	Path      string `json:"path,omitempty" jsonschema:"the directory, the working directory itself by default"`
	Recursive bool   `json:"recursive,omitempty" jsonschema:"list what the subdirectories hold too"`
}

// skipped are the directories a listing or a search does not go into.
var skipped = map[string]bool{".git": true, "node_modules": true}

func (w workdir) listDir(_ context.Context, a listArgs) (string, error) {
	root, err := w.open()
	if err != nil {
		return "", err
	}
	defer root.Close()
	dir := filepath.ToSlash(filepath.Clean(w.rel(a.Path)))
	fsys := root.FS()
	var names []string
	if !a.Recursive {
		entries, err := fs.ReadDir(fsys, dir)
		if err != nil {
			return "", err
		}
		for _, e := range entries {
			names = append(names, e.Name()+slash(e))
		}
	} else {
		err := fs.WalkDir(fsys, dir, func(p string, e fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if p == dir {
				return nil
			}
			if e.IsDir() && skipped[e.Name()] {
				return fs.SkipDir
			}
			rel := strings.TrimPrefix(p, dir+"/")
			if dir == "." {
				rel = p
			}
			names = append(names, rel+slash(e))
			return nil
		})
		if err != nil {
			return "", err
		}
	}
	if len(names) == 0 {
		return "(empty directory)", nil
	}
	return strings.Join(names, "\n"), nil
}

func slash(e fs.DirEntry) string {
	if e.IsDir() {
		return "/"
	}
	return ""
}

type grepArgs struct {
	Pattern string `json:"pattern" jsonschema:"a regular expression, Go syntax"`
	Path    string `json:"path,omitempty" jsonschema:"a file or directory to search, the working directory by default"`
	Glob    string `json:"glob,omitempty" jsonschema:"only files whose name matches, such as *.go"`
}

func (w workdir) grep(ctx context.Context, a grepArgs) (string, error) {
	re, err := regexp.Compile(a.Pattern)
	if err != nil {
		return "", err
	}
	if a.Glob != "" {
		if _, err := path.Match(a.Glob, ""); err != nil {
			return "", fmt.Errorf("glob: %w", err)
		}
	}
	root, err := w.open()
	if err != nil {
		return "", err
	}
	defer root.Close()
	fsys := root.FS()
	start := filepath.ToSlash(filepath.Clean(w.rel(a.Path)))
	var out []string
	full := false
	err = fs.WalkDir(fsys, start, func(p string, e fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if e.IsDir() {
			if p != start && skipped[e.Name()] {
				return fs.SkipDir
			}
			return nil
		}
		if a.Glob != "" {
			if ok, _ := path.Match(a.Glob, e.Name()); !ok {
				return nil
			}
		}
		if info, err := e.Info(); err != nil || !info.Mode().IsRegular() || info.Size() > grepMaxFileSize {
			return nil
		}
		data, err := fs.ReadFile(fsys, p)
		if err != nil || bytes.IndexByte(data, 0) >= 0 {
			return nil
		}
		sc := bufio.NewScanner(bytes.NewReader(data))
		sc.Buffer(make([]byte, 0, 64<<10), grepMaxFileSize)
		for n := 1; sc.Scan(); n++ {
			if re.Match(sc.Bytes()) {
				if len(out) == grepLimit {
					full = true
					return fs.SkipAll
				}
				out = append(out, fmt.Sprintf("%s:%d: %s", p, n, sc.Text()))
			}
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	if len(out) == 0 {
		return "(no match)", nil
	}
	if full {
		out = append(out, fmt.Sprintf("(stopped at %d matches)", grepLimit))
	}
	return strings.Join(out, "\n"), nil
}

type bashArgs struct {
	Command        string `json:"command" jsonschema:"the command, run with sh -c"`
	TimeoutSeconds int    `json:"timeout_seconds,omitempty" jsonschema:"how long it may run, 120 by default, 600 at most"`
}

func (w workdir) bash(ctx context.Context, a bashArgs) (string, error) {
	if strings.TrimSpace(a.Command) == "" {
		return "", errors.New("no command")
	}
	timeout := shellTimeout
	if a.TimeoutSeconds > 0 {
		timeout = min(time.Duration(a.TimeoutSeconds)*time.Second, shellMaxTimeout)
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "sh", "-c", a.Command)
	cmd.Dir = string(w)
	cmd.Stdin = nil
	cmd.WaitDelay = 5 * time.Second
	out, err := cmd.CombinedOutput()
	text := string(out)
	if text == "" {
		text = "(no output)\n"
	} else if !strings.HasSuffix(text, "\n") {
		text += "\n"
	}
	var exit *exec.ExitError
	switch {
	case ctx.Err() == context.DeadlineExceeded:
		return text + fmt.Sprintf("(stopped after %s)", timeout), nil
	case errors.As(err, &exit):
		return text + fmt.Sprintf("(exit code %d)", exit.ExitCode()), nil
	case err != nil:
		return "", err
	}
	return text + "(exit code 0)", nil
}
