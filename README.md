# snaprepo

Flatten a codebase into a single AI-ready snapshot -- handy for handing
an AI agent a whole repo's worth of context in one paste/upload.

## Install (local, editable)

```bash
pipx install --editable .
```

This adds a `snaprepo` command to your PATH, backed by this checkout --
edits to `snaprepo.py` take effect immediately, no reinstall needed.

Don't have `pipx`? On macOS: `brew install pipx && pipx ensurepath`, then
restart your terminal.

Want real token counts (via `tiktoken`) instead of the chars/4 estimate?

```bash
pipx inject snaprepo tiktoken
```

(Optional -- everything else works without it. tiktoken needs a one-time
internet fetch the first time it runs on a machine; if that's blocked,
snaprepo silently falls back to the estimate.)

## Use

```bash
cd ~/some-project
snaprepo                  # interactive
snaprepo --yes            # non-interactive, all defaults
snaprepo --dry-run        # preview only, writes nothing
snaprepo --format markdown  # fenced-code-block .md output instead of .txt
snaprepo --copy           # also copies the snapshot to your clipboard (macOS)
snaprepo --no-gitignore   # ignore .gitignore rules for this run
snaprepo --no-config      # ignore .snaprepo.json for this run
snaprepo --diff           # only what changed since your last commit
snaprepo --diff main      # only what changed vs. main
snaprepo --no-secret-scan # skip the content-based secret check
```

`--diff` requires a git repo and falls back to a full scan (with a
warning) if the directory isn't one, or the ref doesn't exist. Every file
that would otherwise be included also gets content-checked against a
small set of secret-shaped patterns (a live JWT, a PEM key block, an
AWS/OpenAI-style key prefix) -- a match withholds the file and flags it
under SECURITY WARNINGS in the output, and this can't be overridden by
`--include`. It's a heuristic, not a scanner -- always skim the output
before sharing it anywhere.

Output is always `project_snapshot_<UTC timestamp>.txt` (or `.md`),
written into `--output-dir` (default: current directory).

## Per-repo config

Drop a `.snaprepo.json` in the root you scan to store repeatable settings
so you're not retyping flags every run:

```json
{
  "framework": "nextjs",
  "include": ["next.config.js", ".env.example"],
  "exclude": ["*.test.ts", "__mocks__"],
  "output_dir": "snapshots",
  "max_file_size_kb": 2048,
  "gitignore": true,
  "secret_scan": true,
  "format": "markdown"
}
```

All fields optional. Precedence: **CLI flag > `.snaprepo.json` >
interactive prompt > built-in default.**

See `snaprepo --help` for the full flag list, and the module docstring in
`snaprepo.py` for the exact file-inclusion precedence rules.
