package agentkit

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// builtin builds one tool in dir and returns a function that calls it.
func builtin(t *testing.T, dir, name string) func(args string) (string, error) {
	t.Helper()
	tools, err := BuiltinTools(dir, []string{name})
	if err != nil {
		t.Fatal(err)
	}
	return func(args string) (string, error) {
		return tools[0].Run(context.Background(), json.RawMessage(args))
	}
}

func TestBuiltinsAreOffByDefault(t *testing.T) {
	a, err := New(context.Background(), Config{ProviderImpl: &fakeProvider{}})
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Tools()) != 0 {
		t.Fatalf("tools %v", a.Tools())
	}
	a, err = New(context.Background(), Config{ProviderImpl: &fakeProvider{}, Builtins: []string{"read_file", "bash"}, Workdir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(a.Tools(), ","); got != "read_file,bash" {
		t.Fatalf("tools %s", got)
	}
	if _, err := New(context.Background(), Config{ProviderImpl: &fakeProvider{}, Builtins: []string{"rm"}}); err == nil {
		t.Fatal("an unknown built-in was accepted")
	}
}

func TestReadFileNumbersTheLines(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("one\ntwo\nthree\n"), 0o644)
	read := builtin(t, dir, "read_file")
	out, err := read(`{"path":"a.txt"}`)
	if err != nil || out != "1\tone\n2\ttwo\n3\tthree\n" {
		t.Fatalf("%q %v", out, err)
	}
	out, err = read(`{"path":"a.txt","offset":2,"limit":1}`)
	if err != nil || out != "2\ttwo\n(1 lines more, from 3)\n" {
		t.Fatalf("%q %v", out, err)
	}
	// An absolute path inside the directory is taken as it is meant.
	if out, err := read(`{"path":"` + filepath.Join(dir, "a.txt") + `"}`); err != nil || !strings.HasPrefix(out, "1\tone") {
		t.Fatalf("%q %v", out, err)
	}
}

func TestFileToolsStayInTheirDirectory(t *testing.T) {
	parent := t.TempDir()
	dir := filepath.Join(parent, "work")
	os.Mkdir(dir, 0o755)
	os.WriteFile(filepath.Join(parent, "secret"), []byte("no"), 0o644)
	os.Symlink(filepath.Join(parent, "secret"), filepath.Join(dir, "link"))
	read := builtin(t, dir, "read_file")
	write := builtin(t, dir, "write_file")
	for _, args := range []string{`{"path":"../secret"}`, `{"path":"link"}`, `{"path":"` + filepath.Join(parent, "secret") + `"}`} {
		if out, err := read(args); err == nil {
			t.Fatalf("read %s: %q", args, out)
		}
	}
	if _, err := write(`{"path":"../escaped","content":"x"}`); err == nil {
		t.Fatal("wrote outside")
	}
	if _, err := os.Stat(filepath.Join(parent, "escaped")); err == nil {
		t.Fatal("a file was written outside")
	}
}

func TestWriteThenEdit(t *testing.T) {
	dir := t.TempDir()
	write := builtin(t, dir, "write_file")
	edit := builtin(t, dir, "edit_file")
	if _, err := write(`{"path":"sub/dir/f.go","content":"a := 1\nb := 1\n"}`); err != nil {
		t.Fatal(err)
	}
	if _, err := edit(`{"path":"sub/dir/f.go","old_string":"1","new_string":"2"}`); err == nil || !strings.Contains(err.Error(), "2 times") {
		t.Fatalf("an ambiguous edit: %v", err)
	}
	if _, err := edit(`{"path":"sub/dir/f.go","old_string":"zzz","new_string":"2"}`); err == nil {
		t.Fatal("an edit of nothing")
	}
	if _, err := edit(`{"path":"sub/dir/f.go","old_string":"b := 1","new_string":"b := 2"}`); err != nil {
		t.Fatal(err)
	}
	if _, err := edit(`{"path":"sub/dir/f.go","old_string":"a","new_string":"c","replace_all":true}`); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(filepath.Join(dir, "sub/dir/f.go"))
	if string(data) != "c := 1\nb := 2\n" {
		t.Fatalf("%q", data)
	}
}

func TestListDirAndGrep(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "pkg"), 0o755)
	os.MkdirAll(filepath.Join(dir, ".git"), 0o755)
	os.WriteFile(filepath.Join(dir, "pkg/a.go"), []byte("package pkg\nfunc Hello() {}\n"), 0o644)
	os.WriteFile(filepath.Join(dir, "notes.md"), []byte("Hello there\n"), 0o644)
	os.WriteFile(filepath.Join(dir, ".git/config"), []byte("Hello\n"), 0o644)
	os.WriteFile(filepath.Join(dir, "bin"), []byte("Hello\x00"), 0o644)

	list := builtin(t, dir, "list_dir")
	if out, err := list(`{}`); err != nil || out != ".git/\nbin\nnotes.md\npkg/" {
		t.Fatalf("%q %v", out, err)
	}
	if out, err := list(`{"recursive":true}`); err != nil || out != "bin\nnotes.md\npkg/\npkg/a.go" {
		t.Fatalf("%q %v", out, err)
	}
	if out, err := list(`{"path":"pkg"}`); err != nil || out != "a.go" {
		t.Fatalf("%q %v", out, err)
	}

	grep := builtin(t, dir, "grep")
	if out, err := grep(`{"pattern":"Hello"}`); err != nil || out != "notes.md:1: Hello there\npkg/a.go:2: func Hello() {}" {
		t.Fatalf("%q %v", out, err)
	}
	if out, err := grep(`{"pattern":"Hello","glob":"*.go"}`); err != nil || out != "pkg/a.go:2: func Hello() {}" {
		t.Fatalf("%q %v", out, err)
	}
	if out, err := grep(`{"pattern":"Hello","path":"pkg"}`); err != nil || out != "pkg/a.go:2: func Hello() {}" {
		t.Fatalf("%q %v", out, err)
	}
	if _, err := grep(`{"pattern":"("}`); err == nil {
		t.Fatal("a bad pattern was accepted")
	}
}

func TestBash(t *testing.T) {
	dir := t.TempDir()
	bash := builtin(t, dir, "bash")
	out, err := bash(`{"command":"pwd; echo oops >&2; exit 3"}`)
	if err != nil {
		t.Fatal(err)
	}
	real, _ := filepath.EvalSymlinks(dir)
	if !strings.Contains(out, real) || !strings.Contains(out, "oops") || !strings.HasSuffix(out, "(exit code 3)") {
		t.Fatalf("%q", out)
	}
	out, err = bash(`{"command":"sleep 5","timeout_seconds":1}`)
	if err != nil || !strings.Contains(out, "stopped after 1s") {
		t.Fatalf("%q %v", out, err)
	}
}
