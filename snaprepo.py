#!/usr/bin/env python3
"""
snaprepo -- Flatten a codebase into a single AI-ready snapshot.

Walks a project directory, strips out build output, dependency folders,
gitignored files, and config/lockfile noise, and writes everything else
into one file with clear per-file sections -- handy for giving an AI
agent a whole repo's worth of context in one paste/upload instead of many.

USAGE
    Interactive (recommended first run):
        snaprepo

    Non-interactive / scriptable:
        snaprepo --dir ./my-app --framework nextjs \
            --include .ts,.tsx,.md --exclude "*.test.ts,__mocks__" \
            --output-dir ./snapshots --yes

    Preview without writing anything:
        snaprepo --dry-run

    Markdown output instead of plain text:
        snaprepo --format markdown

    Only what changed since your last commit (or vs. a branch):
        snaprepo --diff
        snaprepo --diff main

    See `snaprepo --help` for all flags.

OUTPUT
    The snapshot is always written as:
        project_snapshot_<UTC timestamp>.txt   (or .md with --format markdown)
    e.g. project_snapshot_20260812-184205Z.txt
    The name isn't configurable (so repeated runs never collide or
    silently overwrite each other) -- but --output-dir controls WHERE
    it's written.

CONFIG FILE
    Drop a .snaprepo.json in the root you're scanning to store repeatable
    settings for that repo, so you don't have to retype flags every run:

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

    All fields are optional. Precedence for every setting is:
        CLI flag  >  .snaprepo.json  >  interactive prompt  >  built-in default
    --no-config ignores the file for a single run without deleting it.

PRECEDENCE for which files end up in the snapshot (highest to lowest)
    1. Content that matches a secret-VALUE pattern (a live JWT, a PEM key
       block, an AKIA/sk- style key prefix) -- ALWAYS withheld and listed
       under SECURITY WARNINGS, never overridable, even by --include. See
       SECRET SCANNING below. This is a heuristic, not a guarantee.
    2. Secret-shaped FILENAMES (.env, *.pem, *id_rsa*, ...) -- ALWAYS
       excluded, never overridable. Also a best-effort heuristic, not a
       guarantee -- skim the output before pasting it anywhere, especially
       to a third party.
    3. Your --exclude patterns (CLI + config, combined) -- always win,
       even over your own --include.
    4. Your --include patterns (CLI + config, combined) -- ADDITIVE. They
       don't replace the default whitelist, they extend it, and they can
       also rescue a file from rule 5 below (e.g. adding "next.config.js"
       brings that one file back even though config files are dropped by
       default -- it does NOT narrow the scan down to only that file).
       This rescue power does NOT extend to rule 1 or 2 above -- naming a
       file explicitly still won't surface a secret.
    5. Built-in config/lockfile/binary noise, AND anything matched by a
       .gitignore found in the tree -- dropped by default, rescuable by #4.
    6. The default extension whitelist -- always active. --include adds
       to this; it never replaces it.

Directories (node_modules, dot-dirs, .gitignore matches, your --exclude
patterns) are pruned during the walk, not filtered file-by-file, and are
never rescued by --include -- fast, and avoids listing thousands of
individually-skipped files.

GITIGNORE HANDLING
    Requires the 'pathspec' package (installed automatically alongside
    snaprepo). Each .gitignore found in the tree is scoped to its own
    subtree, same as git. This is a solid best-effort approximation, not
    a byte-for-byte reimplementation of git's ignore resolution -- deeply
    nested override/negation edge cases may resolve slightly differently
    than `git status` would. Disable with --no-gitignore if it ever gets
    in your way.

TOKEN COUNTS
    Uses tiktoken's cl100k_base encoding when available for a real count.
    tiktoken needs a one-time internet fetch the first time it runs on a
    machine (to download its encoding data) -- if that's unavailable
    (offline, restricted network, tiktoken not installed), snaprepo falls
    back to a chars/4 estimate and labels it as such. Either way this is
    a GPT-tokenizer-based count, not Claude's actual tokenizer -- treat
    it as a ballpark, not an exact budget.

SECRET SCANNING
    Every file that would otherwise be included has its full content
    checked against a small set of secret-VALUE patterns -- PEM key
    headers, AKIA-style AWS key IDs, sk-... style provider key prefixes,
    JWT-shaped triples -- separate from the always-on secret-shaped
    FILENAME check (PRECEDENCE #2). A match withholds the file and lists
    it under SECURITY WARNINGS instead of exporting it, even if you
    explicitly --include'd that exact file. This is a heuristic tripwire,
    not a secret scanner: false positives and false negatives are both
    possible -- always skim SECURITY WARNINGS (and the rest of the
    output) before sharing a snapshot anywhere, especially with a third
    party. Disable with --no-secret-scan if it ever misfires on you. Note
    this means --dry-run still reads full file contents (for the scan),
    so it isn't a zero-I/O preview on very large repos.

DIFF MODE
    --diff [REF] (default REF: HEAD) scans only files changed vs. REF,
    plus untracked new files, instead of the whole tree -- a much smaller
    snapshot for "review what I just changed" requests. Requires a git
    repo; falls back to a full scan (with a warning) if the directory
    isn't one, or REF doesn't exist. Deleted files are listed under
    DELETED FILES for visibility -- there's nothing to export for them,
    but a silently smaller snapshot is easy to misread as "nothing
    changed there."
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import pathspec
except ImportError:
    pathspec = None

# --------------------------------------------------------------------------
# Framework registry
# --------------------------------------------------------------------------

FRAMEWORKS = ["nextjs", "astro", "nuxt", "vue", "react", "svelte", "generic"]

FRAMEWORK_LABELS = {
    "nextjs": "Next.js",
    "astro": "Astro",
    "nuxt": "Nuxt",
    "vue": "Vue",
    "react": "React",
    "svelte": "Svelte",
    "generic": "Other / framework-agnostic",
}

# Extra file patterns to drop per framework, on top of the generic rules
# below. Most framework build/cache dirs (.next, .nuxt, .svelte-kit, .astro,
# .output) are already caught by the "skip any dot-directory" rule, so this
# is mainly for stray generated files that don't live in a dot-folder.
FRAMEWORK_EXTRA_EXCLUDES = {
    "nextjs": ["next-env.d.ts"],
    "astro": [],
    "nuxt": [],
    "vue": [],
    "react": [],
    "svelte": [],
    "generic": [],
}

# Best-effort framework auto-detect from package.json. Order matters:
# meta-frameworks are checked before the base library they wrap.
PACKAGE_JSON_FRAMEWORK_HINTS = [
    ("next", "nextjs"),
    ("astro", "astro"),
    ("nuxt", "nuxt"),
    ("@sveltejs/kit", "svelte"),
    ("svelte", "svelte"),
    ("vue", "vue"),
    ("react", "react"),
]

# --------------------------------------------------------------------------
# Built-in ignore rules
# --------------------------------------------------------------------------

# Any directory whose name starts with "." is pruned: .git, .next, .nuxt,
# .svelte-kit, .astro, .output, .vercel, .netlify, .vscode, .idea, .cache,
# .turbo, .github, .husky, .pytest_cache, .ssh, .aws, etc.
# Non-dot build/dependency directories get an explicit list:
BUILTIN_PRUNE_DIR_NAMES = {
    "node_modules", "bower_components", "vendor",
    "dist", "build", "out", "output",
    "coverage", "tmp", "temp", "logs", "log",
    "venv", "env", "__pycache__",
    "target", "bin", "obj",
    "secrets", "credentials",  # extra safety net, see HARD_EXCLUDE_PATTERNS
}

# Config / lockfile / noise patterns. Dropped by default; an explicit
# --include match can rescue individual files from this list (see
# PRECEDENCE above) but --exclude and secrets always win regardless.
BUILTIN_FILE_EXCLUDES = [
    "*.config.js", "*.config.ts", "*.config.mjs", "*.config.cjs",
    "*.config.mts", "*.config.cts",
    "tsconfig*.json", "jsconfig.json",
    ".eslintrc*", ".prettierrc*", ".babelrc*", ".editorconfig",
    ".gitignore", ".gitattributes", ".gitmodules",
    ".npmrc", ".nvmrc", ".prettierignore", ".eslintignore", ".dockerignore",
    "Dockerfile", "Dockerfile.*", "docker-compose*.yml", "docker-compose*.yaml",
    "Makefile",
    "*.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "bun.lockb", "composer.lock", "Cargo.lock", "Gemfile.lock", "poetry.lock",
    "*.log", ".DS_Store", "Thumbs.db",
    "LICENSE", "LICENSE.*",
    ".snaprepo.json",
]

# Secret-shaped files. ALWAYS excluded, never overridable -- see module
# docstring. This is a best-effort net, not a guarantee.
HARD_EXCLUDE_PATTERNS = [
    ".env", ".env.*",
    "*.pem", "*.key", "*.p12", "*.pfx", "*.crt",
    "id_rsa*", "*secret*", "*credential*",
    ".npmrc", ".netrc",
]

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".svg", ".avif",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp4", ".mp3", ".wav", ".avi", ".mov", ".webm",
    ".zip", ".tar", ".gz", ".rar", ".7z", ".br",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".bin", ".wasm",
    ".db", ".sqlite", ".sqlite3", ".class", ".jar",
}

# Always active. --include only ever ADDS to this -- it never replaces it.
DEFAULT_INCLUDE_EXTENSIONS = [
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts",
    ".vue", ".svelte", ".astro",
    ".py", ".go", ".rb", ".php", ".java", ".kt", ".swift",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rs",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".md", ".mdx",
    ".yaml", ".yml",
    ".graphql", ".gql", ".prisma",
    ".sql", ".sh", ".bash", ".txt",
]

# Fenced-code-block language tags for markdown output. Unknown extensions
# just get an unlabeled fence -- still valid markdown, no highlighting.
EXT_TO_LANG = {
    ".ts": "typescript", ".tsx": "tsx", ".js": "javascript", ".jsx": "jsx",
    ".mjs": "javascript", ".cjs": "javascript", ".mts": "typescript", ".cts": "typescript",
    ".vue": "vue", ".svelte": "svelte", ".astro": "astro",
    ".py": "python", ".go": "go", ".rb": "ruby", ".php": "php",
    ".java": "java", ".kt": "kotlin", ".swift": "swift",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".cs": "csharp", ".rs": "rust",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
    ".sass": "sass", ".less": "less",
    ".md": "markdown", ".mdx": "mdx",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json",
    ".graphql": "graphql", ".gql": "graphql", ".prisma": "prisma",
    ".sql": "sql", ".sh": "bash", ".bash": "bash",
}

CONFIG_FILENAME = ".snaprepo.json"
FORMATS = ["text", "markdown"]
OUTPUT_PREFIX = "project_snapshot"
DEFAULT_MAX_FILE_KB = 1024  # skip individual files bigger than this
DUMP_SIGNATURE = "SNAPREPO SNAPSHOT"

# Secret-shaped VALUES, not names -- e.g. a literal JWT or key, not a
# variable named API_KEY (referencing a secret BY NAME is the normal,
# correct way to handle one -- that's the opposite of a leak). See
# SECRET SCANNING in the module docstring.
SECRET_VALUE_REGEX = re.compile(
    r"(BEGIN [A-Z ]*PRIVATE KEY"
    r"|-----BEGIN OPENSSH PRIVATE KEY-----"
    r"|sk-[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"
)


# --------------------------------------------------------------------------
# Pattern helpers
# --------------------------------------------------------------------------

def normalize_include_pattern(token: str) -> str:
    """'.ts' -> '*.ts'; 'ts' -> '*.ts'; 'README.md' -> 'README.md' (literal)."""
    token = token.strip()
    if not token:
        return ""
    if any(ch in token for ch in "*?["):
        return token
    if token.startswith("."):
        return f"*{token}"
    if "/" in token or "\\" in token:
        return token.replace("\\", "/")
    if "." in token:
        return token  # looks like a specific filename
    return f"*.{token}"  # bare word -> treat as an extension


def normalize_exclude_pattern(token: str) -> str:
    """'.log' -> '*.log'; 'test' -> '*test*' (substring); 'a/b' -> 'a/b'."""
    token = token.strip()
    if not token:
        return ""
    if any(ch in token for ch in "*?["):
        return token
    if token.startswith("."):
        return f"*{token}"
    if "/" in token or "\\" in token:
        return token.replace("\\", "/")
    if "." in token:
        return token  # specific filename
    return f"*{token}*"  # bare word -> substring match anywhere in the path


def matches_patterns(name: str, relpath: str, patterns) -> bool:
    if not patterns:
        return False
    name_l = name.lower()
    rel_l = relpath.lower()
    for pat in patterns:
        if not pat:
            continue
        pat_l = pat.lower()
        if fnmatch.fnmatchcase(name_l, pat_l) or fnmatch.fnmatchcase(rel_l, pat_l):
            return True
    return False


def should_prune_dir(dirname: str, relpath: str, user_exclude_patterns) -> bool:
    if dirname.startswith("."):
        return True
    if dirname.lower() in BUILTIN_PRUNE_DIR_NAMES:
        return True
    if matches_patterns(dirname, relpath, user_exclude_patterns):
        return True
    return False


# --------------------------------------------------------------------------
# .gitignore handling
# --------------------------------------------------------------------------

def load_gitignore_spec(dirpath: Path):
    """Parse a .gitignore in dirpath, if present. Returns a PathSpec or None."""
    if pathspec is None:
        return None
    gi_path = dirpath / ".gitignore"
    if not gi_path.is_file():
        return None
    try:
        lines = gi_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    try:
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    except Exception:
        return None


def matches_gitignore(relpath: str, is_dir: bool, gitignore_specs: dict) -> bool:
    """Check relpath against every ancestor directory's .gitignore spec (each
    scoped to its own subtree, same as git). Approximate, not exact: a match
    at ANY level ignores the path -- doesn't replicate git's deeper-overrides
    -shallower negation precedence across separate files."""
    if not gitignore_specs:
        return False
    parts = relpath.split("/")
    for i in range(len(parts)):
        base = "/".join(parts[:i])
        spec = gitignore_specs.get(base)
        if spec is None:
            continue
        sub = "/".join(parts[i:])
        if is_dir:
            sub += "/"
        if spec.match_file(sub):
            return True
    return False


# --------------------------------------------------------------------------
# Config file
# --------------------------------------------------------------------------

def load_config(root: Path) -> dict:
    """Read .snaprepo.json from the scan root, if present. Never raises --
    returns {} and prints a warning on any problem."""
    cfg_path = root / CONFIG_FILENAME
    if not cfg_path.is_file():
        return {}
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: couldn't parse {CONFIG_FILENAME} ({e}) -- ignoring it.")
        return {}
    if not isinstance(data, dict):
        print(f"Warning: {CONFIG_FILENAME} must contain a JSON object -- ignoring it.")
        return {}
    return data


# --------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------

_token_counter = {}


def get_token_counter():
    """Returns (count_fn, label). Tries tiktoken (cl100k_base) once per
    process; falls back permanently to a chars/4 estimate if tiktoken isn't
    installed, or its encoding data can't be fetched (e.g. offline / a
    restricted network -- it needs a one-time download on first use)."""
    if "fn" in _token_counter:
        return _token_counter["fn"], _token_counter["label"]
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        def _count(text: str) -> int:
            return len(enc.encode(text, disallowed_special=()))

        _token_counter["fn"] = _count
        _token_counter["label"] = "tiktoken cl100k_base"
    except Exception:
        def _count(text: str) -> int:
            return len(text) // 4

        _token_counter["fn"] = _count
        _token_counter["label"] = "chars/4 estimate"
    return _token_counter["fn"], _token_counter["label"]


# --------------------------------------------------------------------------
# Misc helpers
# --------------------------------------------------------------------------

def sniff_content_issue(path: Path, sniff_bytes: int = 8000):
    """Peek at a file's bytes; return a skip reason, or None if it looks fine.

    Catches two things with one read: binary content hiding behind a text
    extension, and a previous snaprepo output that happens to still match
    the include filter -- detected by content signature, not filename or
    extension, so it works for both --format text and --format markdown.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(sniff_bytes)
    except OSError:
        return "unreadable"
    if DUMP_SIGNATURE.encode("utf-8") in chunk[:200]:
        return "previous snaprepo output"
    if b"\x00" in chunk:
        return "binary content detected"
    return None


def read_text_best_effort(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"[snaprepo: could not read file -- {e}]"


def human_size(n: float) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def utc_timestamp() -> str:
    """Filesystem-safe UTC timestamp, e.g. 20260812-184205Z (sortable, no colons)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")


def scan_for_secret(path: Path):
    """Full-content check against SECRET_VALUE_REGEX. Returns a skip reason
    string on a match, else None. Heuristic tripwire, not a scanner -- see
    SECRET SCANNING in the module docstring."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if SECRET_VALUE_REGEX.search(text):
        return "matched a secret-pattern tripwire"
    return None


def run_git(root: Path, args):
    """Run a git subcommand scoped to `root`. Returns (ok, lines) -- ok is
    False on any failure (git missing, not a repo, bad ref, ...); lines is
    the non-empty stdout lines on success, else []."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return False, []
    if result.returncode != 0:
        return False, []
    return True, [ln for ln in result.stdout.splitlines() if ln.strip()]


def markdown_fence_for(content: str) -> str:
    """Pick a backtick fence longer than any run already in the content, so
    file contents containing ``` can't break out of their code block."""
    max_run = run = 0
    for ch in content:
        if ch == "`":
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return "`" * max(3, max_run + 1)


def detect_framework(root: Path):
    pkg_path = root / "package.json"
    if not pkg_path.is_file():
        return None
    try:
        data = json.loads(pkg_path.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, OSError):
        return None
    deps = {}
    deps.update(data.get("dependencies") or {})
    deps.update(data.get("devDependencies") or {})
    for pkg_name, framework in PACKAGE_JSON_FRAMEWORK_HINTS:
        if pkg_name in deps:
            return framework
    return None


# --------------------------------------------------------------------------
# Interactive prompts
# --------------------------------------------------------------------------

def prompt_framework(detected):
    print("\nWhich framework is this?")
    for i, key in enumerate(FRAMEWORKS, start=1):
        tag = "   (detected from package.json)" if key == detected else ""
        print(f"  {i}. {FRAMEWORK_LABELS[key]}{tag}")
    default_idx = FRAMEWORKS.index(detected) + 1 if detected else len(FRAMEWORKS)
    raw = input(f"Choice [{default_idx}]: ").strip()
    if not raw:
        return FRAMEWORKS[default_idx - 1]
    try:
        idx = int(raw)
        if 1 <= idx <= len(FRAMEWORKS):
            return FRAMEWORKS[idx - 1]
    except ValueError:
        pass
    print("  Didn't recognize that -- using framework-agnostic mode.")
    return "generic"


# --------------------------------------------------------------------------
# Core walk / collect
# --------------------------------------------------------------------------

def evaluate_file(fname, relpath, abspath, extra_include_patterns, user_exclude_patterns,
                   builtin_excludes, base_include_patterns, gitignore_specs, gitignore_enabled,
                   max_file_size, output_resolved, secret_scan_enabled):
    """The single eligibility gate every candidate file goes through,
    whether it came from the full-tree walk or --diff's git-provided list.
    Returns one of:
        ("self",    None)    -- this IS our own output file, ignore silently
        ("include", size)    -- goes in the snapshot
        ("skip",    reason)  -- left out, filename/pattern/size/content reason
        ("secret",  reason)  -- left out, content matched a secret-VALUE
                                 pattern (never rescuable -- see PRECEDENCE)
    """
    try:
        if abspath.resolve() == output_resolved:
            return "self", None
    except OSError:
        pass

    if matches_patterns(fname, relpath, HARD_EXCLUDE_PATTERNS):
        return "skip", "secret-shaped filename"

    if matches_patterns(fname, relpath, user_exclude_patterns):
        return "skip", "matched your exclude pattern"

    explicit_hit = matches_patterns(fname, relpath, extra_include_patterns)

    if not explicit_hit and matches_patterns(fname, relpath, builtin_excludes):
        return "skip", "config/lockfile noise"

    if not explicit_hit and gitignore_enabled and matches_gitignore(relpath, False, gitignore_specs):
        return "skip", "gitignored"

    if not explicit_hit and abspath.suffix.lower() in BINARY_EXTENSIONS:
        return "skip", "binary file type"

    in_defaults = matches_patterns(fname, relpath, base_include_patterns)
    if not in_defaults and not explicit_hit:
        return "skip", "didn't match include filter"

    try:
        size = abspath.stat().st_size
    except OSError:
        return "skip", "unreadable"
    if size > max_file_size:
        return "skip", f"too large ({human_size(size)})"

    content_issue = sniff_content_issue(abspath)
    if content_issue:
        return "skip", content_issue

    if secret_scan_enabled:
        secret_reason = scan_for_secret(abspath)
        if secret_reason:
            return "secret", secret_reason

    return "include", size


def collect_files(root: Path, framework: str, base_include_patterns,
                   extra_include_patterns, user_exclude_patterns,
                   max_file_size: int, output_path: Path, gitignore_enabled: bool,
                   secret_scan_enabled: bool):
    """Full-tree walk. extra_include_patterns are ADDITIVE to
    base_include_patterns, and are the only patterns with power to rescue a
    file from the builtin config/lockfile/binary/.gitignore excludes (see
    module docstring PRECEDENCE) -- never from a secret-content match."""
    included = []          # (relpath, abspath, size)
    skipped_files = []     # (relpath, reason)
    pruned_dirs = []       # relpath of pruned directories
    security_warnings = [] # relpath of files withheld for a secret-content match

    builtin_excludes = BUILTIN_FILE_EXCLUDES + FRAMEWORK_EXTRA_EXCLUDES.get(framework, [])
    try:
        output_resolved = output_path.resolve()
    except OSError:
        output_resolved = output_path

    gitignore_specs = {}  # dir relpath ("" for root) -> PathSpec
    show_progress = sys.stdout.isatty()  # skip the \r-updating line when piped/redirected
    files_seen = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_p = Path(dirpath)
        dir_relpath = dirpath_p.relative_to(root).as_posix()
        if dir_relpath == ".":
            dir_relpath = ""

        if gitignore_enabled:
            spec = load_gitignore_spec(dirpath_p)
            if spec is not None:
                gitignore_specs[dir_relpath] = spec

        keep = []
        for d in dirnames:
            rel_d = (dirpath_p / d).relative_to(root).as_posix()
            if should_prune_dir(d, rel_d, user_exclude_patterns):
                pruned_dirs.append(rel_d)
                continue
            if gitignore_enabled and matches_gitignore(rel_d, True, gitignore_specs):
                pruned_dirs.append(rel_d)
                continue
            keep.append(d)
        dirnames[:] = keep

        for fname in filenames:
            files_seen += 1
            if show_progress and files_seen % 200 == 0:
                print(f"  ...{files_seen} files scanned", end="\r", flush=True)

            abspath = dirpath_p / fname
            relpath = abspath.relative_to(root).as_posix()

            status, payload = evaluate_file(
                fname, relpath, abspath, extra_include_patterns, user_exclude_patterns,
                builtin_excludes, base_include_patterns, gitignore_specs, gitignore_enabled,
                max_file_size, output_resolved, secret_scan_enabled
            )
            if status == "self":
                continue
            elif status == "include":
                included.append((relpath, abspath, payload))
            elif status == "secret":
                security_warnings.append(relpath)
                skipped_files.append((relpath, payload))
            else:
                skipped_files.append((relpath, payload))

    if show_progress and files_seen >= 200:
        print(" " * 40, end="\r")  # clear the last progress line

    included.sort(key=lambda item: item[0].lower())
    return included, skipped_files, pruned_dirs, security_warnings


def collect_files_diff(root: Path, candidate_relpaths, framework, base_include_patterns,
                        extra_include_patterns, user_exclude_patterns, max_file_size,
                        output_path: Path, secret_scan_enabled: bool):
    """--diff mode: evaluate an exact file list git already gave us (changed
    + untracked), instead of walking the tree. No directory pruning needed
    (there's no walk), and .gitignore is skipped -- git's own output is
    already guaranteed not to include ignored paths."""
    included = []
    skipped_files = []
    security_warnings = []

    builtin_excludes = BUILTIN_FILE_EXCLUDES + FRAMEWORK_EXTRA_EXCLUDES.get(framework, [])
    try:
        output_resolved = output_path.resolve()
    except OSError:
        output_resolved = output_path

    for relpath in candidate_relpaths:
        abspath = root / relpath
        if not abspath.is_file():
            skipped_files.append((relpath, "no longer a regular file on disk"))
            continue
        fname = abspath.name

        status, payload = evaluate_file(
            fname, relpath, abspath, extra_include_patterns, user_exclude_patterns,
            builtin_excludes, base_include_patterns, {}, False,
            max_file_size, output_resolved, secret_scan_enabled
        )
        if status == "self":
            continue
        elif status == "include":
            included.append((relpath, abspath, payload))
        elif status == "secret":
            security_warnings.append(relpath)
            skipped_files.append((relpath, payload))
        else:
            skipped_files.append((relpath, payload))

    included.sort(key=lambda item: item[0].lower())
    return included, skipped_files, security_warnings


def path_in_scope_by_pattern(relpath, extra_include_patterns, user_exclude_patterns,
                              builtin_excludes, base_include_patterns):
    """Pattern-only version of the eligibility check, for deleted files --
    there's no file on disk left to stat/sniff/secret-scan, so this only
    asks "would this path's NAME have been in scope." Used purely to keep
    DELETED FILES from filling up with build-artifact/lockfile noise."""
    fname = relpath.rsplit("/", 1)[-1]
    if matches_patterns(fname, relpath, HARD_EXCLUDE_PATTERNS):
        return False
    if matches_patterns(fname, relpath, user_exclude_patterns):
        return False
    explicit_hit = matches_patterns(fname, relpath, extra_include_patterns)
    if not explicit_hit and matches_patterns(fname, relpath, builtin_excludes):
        return False
    in_defaults = matches_patterns(fname, relpath, base_include_patterns)
    return in_defaults or explicit_hit


# --------------------------------------------------------------------------
# Output builders
# --------------------------------------------------------------------------

def build_summary_fields(root, framework, included, include_tokens, exclude_tokens,
                          total_size, token_count, token_label, diff_ref):
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + " UTC",
        "root": str(root),
        "framework": FRAMEWORK_LABELS.get(framework, framework),
        "file_count": len(included),
        "total_size": human_size(total_size),
        "token_count": token_count,
        "token_label": token_label,
        "include_desc": f"defaults + {', '.join(include_tokens)}" if include_tokens else "defaults (nothing extra)",
        "exclude_desc": ", ".join(exclude_tokens) if exclude_tokens else "(none)",
        "scope_desc": f"changed vs. {diff_ref} (+ untracked)" if diff_ref else "full tree",
    }


def build_header_text(root, framework, included, skipped_files, pruned_dirs,
                       include_tokens, exclude_tokens, total_size, token_count, token_label,
                       security_warnings=None, diff_ref=None, deleted_files=None):
    security_warnings = security_warnings or []
    deleted_files = deleted_files or []
    f = build_summary_fields(root, framework, included, include_tokens, exclude_tokens,
                              total_size, token_count, token_label, diff_ref)
    header = []
    header.append("=" * 80)
    header.append(DUMP_SIGNATURE)
    header.append("=" * 80)
    header.append(f"Generated:      {f['generated']}")
    header.append(f"Root:           {f['root']}")
    header.append(f"Framework:      {f['framework']}")
    header.append(f"Scope:          {f['scope_desc']}")
    header.append(f"Files included: {f['file_count']}  ({f['total_size']}, ~{f['token_count']:,} tokens [{f['token_label']}])")
    header.append(f"Include filter: {f['include_desc']}")
    header.append(f"Extra excludes: {f['exclude_desc']}")
    header.append("")
    header.append(f"-- FILE INDEX ({len(included)}) --")
    for relpath, _, size in included:
        header.append(f"  {relpath}  ({human_size(size)})")

    uniq_pruned = sorted(set(pruned_dirs))
    if uniq_pruned:
        header.append("")
        header.append(f"-- SKIPPED DIRECTORIES ({len(uniq_pruned)}) --")
        for d in uniq_pruned[:50]:
            header.append(f"  {d}/")
        if len(uniq_pruned) > 50:
            header.append(f"  ... and {len(uniq_pruned) - 50} more")

    if deleted_files:
        header.append("")
        header.append(f"-- DELETED FILES ({len(deleted_files)}) --")
        header.append(f"  In scope but deleted since {diff_ref} -- nothing to export, listed for visibility:")
        for d in deleted_files:
            header.append(f"  {d}")

    if security_warnings:
        header.append("")
        header.append(f"-- SECURITY WARNINGS ({len(security_warnings)}) --")
        header.append("  Matched a secret-pattern tripwire and were NOT exported. Heuristic, not")
        header.append("  a scanner -- review these manually before sharing this snapshot:")
        for relpath in security_warnings:
            header.append(f"  {relpath}")

    if skipped_files:
        header.append("")
        header.append(f"-- SKIPPED FILES ({len(skipped_files)}) --")
        for relpath, reason in skipped_files[:100]:
            header.append(f"  {relpath}  [{reason}]")
        if len(skipped_files) > 100:
            header.append(f"  ... and {len(skipped_files) - 100} more")

    return header


def build_header_markdown(root, framework, included, skipped_files, pruned_dirs,
                           include_tokens, exclude_tokens, total_size, token_count, token_label,
                           security_warnings=None, diff_ref=None, deleted_files=None):
    security_warnings = security_warnings or []
    deleted_files = deleted_files or []
    f = build_summary_fields(root, framework, included, include_tokens, exclude_tokens,
                              total_size, token_count, token_label, diff_ref)
    lines = []
    lines.append(f"<!-- {DUMP_SIGNATURE} -->")
    lines.append("# Project Snapshot")
    lines.append("")
    lines.append(f"- **Generated:** {f['generated']}")
    lines.append(f"- **Root:** `{f['root']}`")
    lines.append(f"- **Framework:** {f['framework']}")
    lines.append(f"- **Scope:** {f['scope_desc']}")
    lines.append(f"- **Files included:** {f['file_count']} ({f['total_size']}, ~{f['token_count']:,} tokens [{f['token_label']}])")
    lines.append(f"- **Include filter:** {f['include_desc']}")
    lines.append(f"- **Extra excludes:** {f['exclude_desc']}")
    lines.append("")
    lines.append(f"## File Index ({len(included)})")
    lines.append("")
    for relpath, _, size in included:
        lines.append(f"- `{relpath}` ({human_size(size)})")

    uniq_pruned = sorted(set(pruned_dirs))
    if uniq_pruned:
        lines.append("")
        lines.append(f"## Skipped Directories ({len(uniq_pruned)})")
        lines.append("")
        for d in uniq_pruned[:50]:
            lines.append(f"- `{d}/`")
        if len(uniq_pruned) > 50:
            lines.append(f"- ... and {len(uniq_pruned) - 50} more")

    if deleted_files:
        lines.append("")
        lines.append(f"## Deleted Files ({len(deleted_files)})")
        lines.append("")
        lines.append(f"In scope but deleted since {diff_ref} -- nothing to export, listed for visibility:")
        lines.append("")
        for d in deleted_files:
            lines.append(f"- `{d}`")

    if security_warnings:
        lines.append("")
        lines.append(f"## Security Warnings ({len(security_warnings)})")
        lines.append("")
        lines.append("Matched a secret-pattern tripwire and were **not** exported. Heuristic, not")
        lines.append("a scanner -- review these manually before sharing this snapshot:")
        lines.append("")
        for relpath in security_warnings:
            lines.append(f"- `{relpath}`")

    if skipped_files:
        lines.append("")
        lines.append(f"## Skipped Files ({len(skipped_files)})")
        lines.append("")
        for relpath, reason in skipped_files[:100]:
            lines.append(f"- `{relpath}` — {reason}")
        if len(skipped_files) > 100:
            lines.append(f"- ... and {len(skipped_files) - 100} more")

    return lines


def write_output(root, framework, included, skipped_files, pruned_dirs, output_path,
                  include_tokens, exclude_tokens, fmt: str,
                  security_warnings=None, diff_ref=None, deleted_files=None):
    counter, token_label = get_token_counter()

    file_blocks = []
    total_chars = 0
    token_count = 0
    for relpath, abspath, _size in included:
        content = read_text_best_effort(abspath)
        total_chars += len(content)
        token_count += counter(content)
        file_blocks.append((relpath, content))

    total_size = sum(size for _, _, size in included)

    if fmt == "markdown":
        header = build_header_markdown(root, framework, included, skipped_files, pruned_dirs,
                                        include_tokens, exclude_tokens, total_size, token_count, token_label,
                                        security_warnings, diff_ref, deleted_files)
    else:
        header = build_header_text(root, framework, included, skipped_files, pruned_dirs,
                                    include_tokens, exclude_tokens, total_size, token_count, token_label,
                                    security_warnings, diff_ref, deleted_files)
    header.append("")
    header.append("=" * 80 if fmt != "markdown" else "---")
    header.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n".join(header))
        out.write("\n")
        for relpath, content in file_blocks:
            if fmt == "markdown":
                lang = EXT_TO_LANG.get(Path(relpath).suffix.lower(), "")
                fence = markdown_fence_for(content)
                out.write(f"## `{relpath}`\n\n")
                out.write(f"{fence}{lang}\n")
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
                out.write(f"{fence}\n\n")
            else:
                out.write("=" * 80 + "\n")
                out.write(f"FILE: {relpath}\n")
                out.write("=" * 80 + "\n")
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
                out.write("\n")

    return total_size, token_count, token_label


def copy_to_clipboard(output_path: Path) -> bool:
    """Best-effort clipboard copy. macOS (pbcopy) only for now -- returns
    False (and prints why) if that's not available on this machine."""
    if sys.platform != "darwin":
        print("(--copy skipped: clipboard copy only supports macOS/pbcopy right now)")
        return False
    try:
        data = output_path.read_bytes()
        subprocess.run(["pbcopy"], input=data, check=True)
        return True
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"(--copy failed: {e})")
        return False


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        prog="snaprepo",
        description="Flatten a codebase into a single AI-ready snapshot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  snaprepo\n"
            "  snaprepo --dir ./app --framework nextjs --yes\n"
            "  snaprepo --include next.config.js,.env.example --yes\n"
            "  snaprepo --dry-run\n"
            "  snaprepo --format markdown\n"
            "  snaprepo --no-gitignore --no-config\n"
            "  snaprepo --diff              # everything changed since your last commit\n"
            "  snaprepo --diff main         # everything changed vs. main\n"
            "  snaprepo --no-secret-scan\n"
            "  snaprepo --copy   # also puts the snapshot on your clipboard (macOS)\n"
        ),
    )
    parser.add_argument("--dir", default=None, help="Root directory to scan (default: prompt, falls back to '.')")
    parser.add_argument("--framework", choices=FRAMEWORKS, default=None, help="Skip the framework prompt")
    parser.add_argument("--diff", nargs="?", const="HEAD", default=None, metavar="REF",
                         help="Only files changed vs. REF (default HEAD) plus untracked files, "
                              "instead of the whole tree. Requires a git repo.")
    parser.add_argument("--include", default=None,
                         help="Comma-delimited patterns to ADD on top of the default include set "
                              "(does not replace it -- skips the prompt)")
    parser.add_argument("--exclude", default=None, help="Comma-delimited exclude patterns (skips that prompt)")
    parser.add_argument("--output-dir", default=None,
                         help="Directory to write the snapshot into (default: prompt, falls back to '.'). "
                              f"Filename is always {OUTPUT_PREFIX}_<UTC timestamp>.txt (or .md)")
    parser.add_argument("--format", choices=FORMATS, default=None,
                         help="Output format (default: text, or .snaprepo.json's 'format')")
    parser.add_argument("--max-file-size-kb", type=int, default=None,
                         help=f"Skip individual files larger than this (default: {DEFAULT_MAX_FILE_KB} KB)")
    parser.add_argument("--no-gitignore", action="store_true",
                         help="Don't apply .gitignore rules found in the tree")
    parser.add_argument("--no-config", action="store_true",
                         help=f"Ignore {CONFIG_FILENAME} in the scan root for this run")
    parser.add_argument("--no-secret-scan", action="store_true",
                         help="Don't content-scan files for secret-shaped values before including them")
    parser.add_argument("--dry-run", action="store_true",
                         help="Preview which files would be included -- writes nothing to disk")
    parser.add_argument("--copy", action="store_true",
                         help="Also copy the finished snapshot to the clipboard (macOS only for now)")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip all interactive prompts; use flags/config/defaults")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  SNAPREPO -- flatten a codebase into one AI-ready snapshot")
    print("=" * 60)

    try:
        if args.dir:
            root = Path(args.dir).expanduser().resolve()
        elif args.yes:
            root = Path(".").resolve()
        else:
            raw = input("\nRoot directory to scan [.]: ").strip()
            root = Path(raw or ".").expanduser().resolve()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)

    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.")
        sys.exit(1)

    config = {} if args.no_config else load_config(root)
    if config:
        print(f"Using settings from {CONFIG_FILENAME} (CLI flags still override them).")

    detected = detect_framework(root)

    try:
        if args.framework:
            framework = args.framework
        elif config.get("framework") in FRAMEWORKS:
            framework = config["framework"]
        elif args.yes:
            framework = detected or "generic"
        else:
            framework = prompt_framework(detected)

        config_include = [str(t) for t in config.get("include", [])]
        if args.include is not None:
            cli_include = [t.strip() for t in args.include.split(",") if t.strip()]
        elif config_include or args.yes:
            cli_include = []
        else:
            raw = input(
                "\nAny extra file types/patterns to include ON TOP OF the smart defaults,\n"
                "comma-delimited (e.g. next.config.js,.env.example -- bare filenames also\n"
                "work and will override the default config/lockfile exclusions for just\n"
                "that file). Leave blank to just use the defaults: "
            ).strip()
            cli_include = [t.strip() for t in raw.split(",") if t.strip()]
        include_tokens = config_include + cli_include

        config_exclude = [str(t) for t in config.get("exclude", [])]
        if args.exclude is not None:
            cli_exclude = [t.strip() for t in args.exclude.split(",") if t.strip()]
        elif config_exclude or args.yes:
            cli_exclude = []
        else:
            raw = input(
                "\nAlso exclude these files/folders, comma-delimited\n"
                "(e.g. *.test.ts,__mocks__,legacy) -- leave blank for none: "
            ).strip()
            cli_exclude = [t.strip() for t in raw.split(",") if t.strip()]
        exclude_tokens = config_exclude + cli_exclude

        if args.output_dir:
            output_dir_raw = args.output_dir
        elif config.get("output_dir"):
            output_dir_raw = str(config["output_dir"])
        elif args.yes:
            output_dir_raw = "."
        else:
            output_dir_raw = input(
                f"\nDirectory to save the snapshot in [.] "
                f"(filename is always {OUTPUT_PREFIX}_<UTC timestamp>.txt): "
            ).strip() or "."
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)

    if args.format:
        fmt = args.format
    elif config.get("format") in FORMATS:
        fmt = config["format"]
    else:
        fmt = "text"

    if args.no_gitignore:
        gitignore_enabled = False
    else:
        gitignore_enabled = bool(config.get("gitignore", True))
    if gitignore_enabled and pathspec is None:
        print("(.gitignore-aware pruning wants the 'pathspec' package, which isn't installed --"
              " skipping it for this run. `pip install pathspec` to enable it.)")
        gitignore_enabled = False

    if args.max_file_size_kb is not None:
        max_file_kb = args.max_file_size_kb
    elif "max_file_size_kb" in config:
        max_file_kb = config["max_file_size_kb"]
    else:
        max_file_kb = DEFAULT_MAX_FILE_KB

    if args.no_secret_scan:
        secret_scan_enabled = False
    else:
        secret_scan_enabled = bool(config.get("secret_scan", True))

    base_include_patterns = [f"*{ext}" for ext in DEFAULT_INCLUDE_EXTENSIONS]
    extra_include_patterns = [normalize_include_pattern(t) for t in include_tokens]
    exclude_patterns = [normalize_exclude_pattern(t) for t in exclude_tokens]

    output_dir_path = Path(output_dir_raw).expanduser()
    if not output_dir_path.is_absolute():
        output_dir_path = Path.cwd() / output_dir_path
    ext = "md" if fmt == "markdown" else "txt"
    output_filename = f"{OUTPUT_PREFIX}_{utc_timestamp()}.{ext}"
    output_path = output_dir_path / output_filename

    max_size = max_file_kb * 1024

    # -----------------------------------------------------------------
    # --diff setup: resolve to an exact candidate file list, or fall back
    # to a full scan (with a warning) if this isn't a usable git repo/ref.
    # -----------------------------------------------------------------
    diff_ref = None
    diff_candidates = None
    deleted_in_scope = []

    if args.diff is not None:
        requested_ref = args.diff
        is_repo, _ = run_git(root, ["rev-parse", "--is-inside-work-tree"])
        if not is_repo:
            print(f"(--diff ignored: '{root}' isn't a git repo -- scanning the full tree instead.)")
        else:
            ref_ok, _ = run_git(root, ["rev-parse", "--verify", requested_ref])
            if not ref_ok:
                print(f"(--diff ignored: git ref '{requested_ref}' not found -- scanning the full tree instead.)")
            else:
                diff_ref = requested_ref
                _, changed = run_git(root, ["diff", "--name-only", "--diff-filter=ACMR", diff_ref])
                _, untracked = run_git(root, ["ls-files", "--others", "--exclude-standard"])
                diff_candidates = sorted(set(changed) | set(untracked))
                _, deleted_raw = run_git(root, ["diff", "--name-only", "--diff-filter=D", diff_ref])
                builtin_excludes_for_deleted = BUILTIN_FILE_EXCLUDES + FRAMEWORK_EXTRA_EXCLUDES.get(framework, [])
                deleted_in_scope = [
                    d for d in deleted_raw
                    if path_in_scope_by_pattern(d, extra_include_patterns, exclude_patterns,
                                                 builtin_excludes_for_deleted, base_include_patterns)
                ]
                if deleted_in_scope:
                    print(f"\nNote: {len(deleted_in_scope)} file(s) deleted since {diff_ref} "
                          f"(not exported -- see DELETED FILES in the output):")
                    for d in deleted_in_scope:
                        print(f"  - {d}")

    if diff_candidates is not None:
        print(f"\nScanning {root} (--diff {diff_ref}) ...")
        included, skipped_files, security_warnings = collect_files_diff(
            root, diff_candidates, framework, base_include_patterns, extra_include_patterns,
            exclude_patterns, max_size, output_path, secret_scan_enabled
        )
        pruned_dirs = []
    else:
        print(f"\nScanning {root} ...")
        included, skipped_files, pruned_dirs, security_warnings = collect_files(
            root, framework, base_include_patterns, extra_include_patterns,
            exclude_patterns, max_size, output_path, gitignore_enabled, secret_scan_enabled
        )

    if not included:
        if diff_candidates is not None and not deleted_in_scope:
            print("\nNothing changed. Nothing to snapshot.")
        else:
            print("\nNo files matched. Try adding some --include patterns.")
        sys.exit(0)

    if args.dry_run:
        total_size = sum(size for _, _, size in included)
        est_tokens = total_size // 4  # byte-based estimate; no file reads in dry-run mode
        if fmt == "markdown":
            header = build_header_markdown(root, framework, included, skipped_files, pruned_dirs,
                                            include_tokens, exclude_tokens, total_size, est_tokens,
                                            "byte-based estimate, dry-run",
                                            security_warnings, diff_ref, deleted_in_scope)
        else:
            header = build_header_text(root, framework, included, skipped_files, pruned_dirs,
                                        include_tokens, exclude_tokens, total_size, est_tokens,
                                        "byte-based estimate, dry-run",
                                        security_warnings, diff_ref, deleted_in_scope)
        print("\n" + "\n".join(header))
        print(f"\n(dry run -- nothing written; would have saved to {output_path})")
        return

    total_size, token_count, token_label = write_output(
        root, framework, included, skipped_files, pruned_dirs,
        output_path, include_tokens, exclude_tokens, fmt,
        security_warnings, diff_ref, deleted_in_scope
    )

    print(f"\nDone -- {len(included)} files written to {output_path}")
    print(f"Size: {human_size(total_size)}  |  ~{token_count:,} tokens [{token_label}]")
    if skipped_files:
        print(f"({len(skipped_files)} files skipped -- see the index at the top of the output for why)")
    if pruned_dirs:
        print(f"({len(set(pruned_dirs))} directories skipped entirely, e.g. node_modules/.git/build)")
    if security_warnings:
        print(f"⚠ {len(security_warnings)} file(s) matched a secret-pattern tripwire and were withheld "
              f"-- see SECURITY WARNINGS in {output_path}.")

    if args.copy:
        if copy_to_clipboard(output_path):
            print("Copied to clipboard.")


if __name__ == "__main__":
    main()
