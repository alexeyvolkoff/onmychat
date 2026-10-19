"""On My Chat — Craft editors service (WordCraft / GridCraft / DeckCraft via their CLIs).

Bridges OMD documents to the ArtCraft command-line tools so the MCP agent can
read, create and edit REAL .docx / .xlsx / .pptx files with full fidelity.

Flow per call: download from OMD → run the CLI in a private tmpdir → upload back.
The CLIs are confined to that tmpdir by construction: `edit_document` allowlists
command ids and parses each command's JSON to reject string parameters that name
a filesystem path (`path`, `original`, `file`, …) — a raw string scan would be
fooled by JSON escapes (`{"path": …}`), while the engine decodes them. The
`--automation-read-root/--automation-write-root` roots exist only on the `mcp`
subcommand (see `_run`), which is why this validation lives here.

Kind → CLI mapping (binaries are looked up in $CRAFT_CLI_DIR or the repos'
target/release next to this checkout):
  doc   → wordcraft-cli   (.docx .odt .rtf .doc .html .md .tex .txt)
  sheet → gridcraft-cli   (.xlsx .csv .tsv)
  deck  → deckcraft-cli   (.pptx)
"""

import base64
import json
import os
import shutil
import socket
import subprocess
import tempfile
import zipfile

from typing import Optional

# Where the release binaries live. Override with CRAFT_CLI_DIR.
_REPO_ROOTS = {
    "doc": "/home/alexey/projects/wordcraft/target/release/wordcraft-cli",
    "sheet": "/home/alexey/projects/gridcraft/target/release/gridcraft-cli",
    "deck": "/home/alexey/projects/deckcraft/target/release/deckcraft-cli",
}

DOC_EXTS = {".docx", ".odt", ".rtf", ".doc", ".html", ".htm", ".md", ".tex", ".txt"}
SHEET_EXTS = {".xlsx", ".csv", ".tsv"}
DECK_EXTS = {".pptx"}

# Command prefixes the agent may run through edit_document. Everything that
# touches paths (file.*, insert.picture with a path, mail merge, templates…) is
# outside the allowlist; the downloads/uploads are done by this service itself.
EDIT_ALLOWLIST_PREFIXES = (
    "text.", "caret.", "select.", "format.", "style.", "list.", "table.", "paragraph.",
    "page.", "section.", "insert.symbol", "insert.break", "insert.datetime", "insert.field",
    "reference.", "review.", "font.", "clipboard.", "document.", "edit.",
    "home.", "cell.", "row.", "column.", "sheet.", "range.", "formula.", "data.", "sort.",
    "slide.", "shape.", "layout.", "theme.", "transition.", "animation.",
)
MAX_DOC_BYTES = 60 * 1024 * 1024
DEFAULT_TIMEOUT = 90
MAX_EDIT_TIMEOUT = 600

# Parameter keys whose STRING value names a filesystem path — engine commands
# that read a file by themselves while staying inside the id allowlist
# (review.compare/combine `original`, font.add `path`, gridcraft data.* `path`…).
# Only non-empty strings count: document.paragraph / shape.connect use `path`
# as a story/geometry list, which is not a file.
FS_PARAM_KEYS = {"path", "original", "file", "files", "filename", "filepath",
                 "directory", "folder"}


def _fs_param_key(value) -> Optional[str]:
    """First denylisted key holding a non-empty string, searched recursively."""
    if isinstance(value, dict):
        for k, v in value.items():
            if str(k).lower() in FS_PARAM_KEYS and isinstance(v, str) and v.strip():
                return str(k)
            hit = _fs_param_key(v)
            if hit:
                return hit
    elif isinstance(value, list):
        for item in value:
            hit = _fs_param_key(item)
            if hit:
                return hit
    return None


def _cli_path(kind: str) -> Optional[str]:
    override_dir = os.environ.get("CRAFT_CLI_DIR")
    name = {"doc": "wordcraft-cli", "sheet": "gridcraft-cli", "deck": "deckcraft-cli"}[kind]
    if override_dir:
        p = os.path.join(override_dir, name)
        if os.path.isfile(p):
            return p
    p = _REPO_ROOTS[kind]
    return p if os.path.isfile(p) else None


def node_name() -> str:
    """This node's name: $CRAFT_NODE_NAME, then the onmydisk config's hostname
    (`hostname=beelink` in /etc/onmydisk/onmydisk.conf.user), then the OS hostname."""
    env = os.environ.get("CRAFT_NODE_NAME")
    if env:
        return env.strip().strip("/")
    try:
        conf = "/etc/onmydisk/onmydisk.conf.user"
        if os.path.isfile(conf):
            import configparser
            cp = configparser.ConfigParser()
            cp.read(conf)
            host = cp.get("Settings", "hostname", fallback="").strip()
            if host:
                return host
    except Exception:
        pass
    return socket.gethostname().split(".")[0]


def storage_root() -> str:
    """Local folder that backs this node's shares (onmychat config STORAGE_ROOT)."""
    try:
        from config import SETTINGS
        root = SETTINGS.get("STORAGE_ROOT", "")
        if root:
            return os.path.realpath(root)
    except Exception:
        pass
    return os.path.expanduser("~")


def resolve_local_path(ctx, path: str) -> Optional[str]:
    """The LOCAL case: a PRIVATE context, the first path section names THIS node,
    and the file lives under this node's STORAGE_ROOT. Returns the absolute local
    path, or None when the gateway/P2P stack must be used (another node, a guest,
    a wider path). OMD_CRAFT_ALLOW_LOCAL=0 disables the shortcut."""
    if os.environ.get("OMD_CRAFT_ALLOW_LOCAL") == "0":
        return None
    # Owner-only. api.trusted_private_access marks trusted requests
    # private_mode=True (tunnel role owner, paired DID, validated owner); a
    # client-supplied omd_key is NOT proof of ownership — api.get_ctx builds it
    # straight from request headers/payload — so it must not open the disk.
    if getattr(ctx, "private_mode", False) is not True:
        return None
    parts = [p for p in (path or "").split("/") if p]
    if not parts:
        return None
    if parts[0].lower() != node_name().lower():
        return None
    # A NODE OWNER (tunnel role=owner / paired DID, api.trusted_owner_access)
    # reads anywhere on their own node inside STORAGE_ROOT — bouncing through
    # the gateway for the owner's own disk is pointless. Anyone else who got
    # private mode (a pm=1 shared guest) stays inside the storage subtree they
    # declared: the gateway enforces a finer ACL there, and this shortcut skips it.
    if getattr(ctx, "trusted_owner", False) is not True:
        storage = (getattr(ctx, "storage", "") or "").strip("/")
        if storage:
            sections = [s for s in storage.split("/") if s]
            if parts[0].lower() != sections[0].lower():
                return None
            tail = [s.lower() for s in sections[1:]]
            if [p.lower() for p in parts[1:1 + len(tail)]] != tail:
                return None
    root = storage_root()
    rest = parts[1:]
    # Two layouts exist in the wild: shares directly under STORAGE_ROOT
    # (/beelink/Data/x → STORAGE_ROOT/Data/x) and the node folder itself
    # (STORAGE_ROOT/beelink/Data/x). A path must resolve to exactly one of them.
    candidates = [os.path.realpath(os.path.join(root, *rest)), os.path.realpath(os.path.join(root, parts[0], *rest))]
    hits = [c for c in candidates if os.path.exists(c)]
    if len(hits) == 1:
        resolved = hits[0]
    elif not hits:
        # A new file: resolvable only when exactly one candidate folder exists.
        existing_parents = [c for c in candidates if os.path.isdir(os.path.dirname(c))]
        if len(existing_parents) != 1:
            return None
        resolved = existing_parents[0]
    else:
        return None
    if not resolved.startswith(root + os.sep):
        return None  # never escape STORAGE_ROOT
    return resolved


def kind_for_filename(filename: str) -> Optional[str]:
    ext = os.path.splitext(filename)[1].lower()
    if ext in DOC_EXTS:
        return "doc"
    if ext in SHEET_EXTS:
        return "sheet"
    if ext in DECK_EXTS:
        return "deck"
    return None


def available_kinds() -> list:
    return [k for k in ("doc", "sheet", "deck") if _cli_path(k)]


def _run(cli: str, args: list, workdir: str, timeout: int = DEFAULT_TIMEOUT) -> subprocess.CompletedProcess:
    """Run a CLI in the private workdir. NOTE: the `--automation-*` roots only
    exist on the `mcp` subcommand; the direct subcommands (text/convert/run/cat)
    don't take them. Confinement here is by construction: the only agent inputs
    are the document bytes (downloaded by this service) and allowlisted commands
    without any path parameters — this service owns all the file names."""
    return subprocess.run([cli, *args], cwd=workdir, capture_output=True, text=True, timeout=timeout)


def _check(data: bytes) -> None:
    if not data:
        raise ValueError("empty document data")
    if len(data) > MAX_DOC_BYTES:
        raise ValueError(f"document too large: {len(data)} bytes (max {MAX_DOC_BYTES})")


def _tmpfile(workdir: str, name: str) -> str:
    return os.path.join(workdir, os.path.basename(name))


def _assert_zip(data: bytes, what: str) -> None:
    """A docx/xlsx/pptx is a zip: catch a CLI that wrote garbage before uploading."""
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(data)) as z:
            bad = z.testzip()
            if bad is not None:
                raise ValueError(f"{what}: corrupt entry {bad!r}")
    except zipfile.BadZipFile as e:
        raise ValueError(f"{what}: not a valid OOXML package ({e})")


def read_document_text(kind: str, data: bytes, source_name: str) -> str:
    """Plain text: doc → `text`, sheet → `cat --csv`, deck → outline via convert .md."""
    _check(data)
    cli = _cli_path(kind)
    if not cli:
        raise RuntimeError(f"{kind} CLI is not available on this node")
    with tempfile.TemporaryDirectory(prefix="omd-craft-") as wd:
        src = _tmpfile(wd, source_name or "doc")
        with open(src, "wb") as f:
            f.write(data)
        if kind == "doc":
            args = ["text", src]
        elif kind == "sheet":
            args = ["cat", src, "--csv"]
        else:
            out_md = _tmpfile(wd, "out.md")
            r0 = _run(cli, ["convert", src, out_md], wd)
            if r0.returncode != 0:
                raise RuntimeError(f"craft outline failed: {r0.stderr.strip()[:500]}")
            with open(out_md, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        r = _run(cli, args, wd)
        if r.returncode != 0:
            raise RuntimeError(f"craft text failed: {r.stderr.strip()[:500]}")
        return r.stdout


def read_document_markdown(kind: str, data: bytes, source_name: str) -> str:
    """Markdown view of a document (doc only): convert to .md and read it."""
    _check(data)
    cli = _cli_path(kind)
    if not cli:
        raise RuntimeError(f"{kind} CLI is not available on this node")
    if kind != "doc":
        raise ValueError("markdown export is only available for text documents (kind=doc)")
    with tempfile.TemporaryDirectory(prefix="omd-craft-") as wd:
        src = _tmpfile(wd, source_name or "in.docx")
        out = _tmpfile(wd, "out.md")
        with open(src, "wb") as f:
            f.write(data)
        r = _run(cli, ["convert", src, out], wd)
        if r.returncode != 0:
            raise RuntimeError(f"craft convert failed: {r.stderr.strip()[:500]}")
        with open(out, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


def inspect_document(kind: str, data: bytes, source_name: str) -> str:
    """Structure JSON of a document (doc: `inspect`)."""
    _check(data)
    cli = _cli_path(kind)
    if not cli:
        raise RuntimeError(f"{kind} CLI is not available on this node")
    with tempfile.TemporaryDirectory(prefix="omd-craft-") as wd:
        src = _tmpfile(wd, source_name or "in.docx")
        with open(src, "wb") as f:
            f.write(data)
        args = ["inspect", src] if kind == "doc" else ["cat", src]
        r = _run(cli, args, wd)
        if r.returncode != 0:
            raise RuntimeError(f"craft inspect failed: {r.stderr.strip()[:500]}")
        return r.stdout


def create_document(kind: str, target_name: str, content: str, title: Optional[str] = None) -> bytes:
    """Create a document from agent text. kind=doc: content is Markdown (with
    # headings, lists, **bold** — the CLI converts md → docx/odt/rtf/pdf/html by
    the target extension). kind=sheet: content is CSV. kind=deck: content is an
    outline (one line per slide, indent for bullets)."""
    _check(content.encode("utf-8"))
    cli = _cli_path(kind)
    if not cli:
        raise RuntimeError(f"{kind} CLI is not available on this node")
    target_name = os.path.basename(target_name) or "document.docx"
    ext = os.path.splitext(target_name)[1].lower() or (".docx" if kind == "doc" else ".xlsx" if kind == "sheet" else ".pptx")
    with tempfile.TemporaryDirectory(prefix="omd-craft-") as wd:
        if kind == "doc":
            src = _tmpfile(wd, "content.md")
            out = _tmpfile(wd, "out" + ext)
            with open(src, "w", encoding="utf-8") as f:
                f.write(content)
            r = _run(cli, ["convert", src, out], wd)
            if r.returncode != 0:
                raise RuntimeError(f"craft create failed: {r.stderr.strip()[:800]}")
        elif kind == "sheet":
            src = _tmpfile(wd, "content.csv")
            out = _tmpfile(wd, "out" + ext)  # .xlsx / .csv / .tsv — the target extension wins
            with open(src, "w", encoding="utf-8") as f:
                f.write(content)
            r = _run(cli, ["convert", src, out], wd)
            if r.returncode != 0:
                raise RuntimeError(f"craft create (sheet) failed: {r.stderr.strip()[:800]}")
        else:
            # Deck: build the deck from the outline — one non-empty line = one
            # slide title (run starts from an empty presentation when no --in
            # and no --sample are given). Bullets can be added later with
            # edit_document (slide/text commands).
            out = _tmpfile(wd, "out.pptx")
            args = ["run"]
            for line in content.splitlines():
                t = line.strip()
                if not t:
                    continue
                if t.startswith("# "):
                    t = t[2:]
                args += ["--cmd", "slide.new=" + json.dumps({"title": t})]
            if len(args) == 1:
                raise ValueError("empty outline: give one non-empty line per slide")
            args += ["--save", out]
            r = _run(cli, args, wd)
            if r.returncode != 0:
                raise RuntimeError(f"craft create (deck) failed: {r.stderr.strip()[:800]}")
        with open(out, "rb") as f:
            data = f.read()
    if ext in (".docx", ".xlsx", ".pptx"):
        _assert_zip(data, target_name)
    return data


def edit_document(kind: str, data: bytes, source_name: str, commands: list, author: str = "OMD Agent",
                  out_name: Optional[str] = None) -> bytes:
    """Run engine commands headlessly on a downloaded document and return the
    saved bytes. commands: list of `id={"json": params}` strings, e.g.
    'format.bold={}'; confined by an allowlist (no path-carrying commands)."""
    _check(data)
    cli = _cli_path(kind)
    if not cli:
        raise RuntimeError(f"{kind} CLI is not available on this node")
    if not commands:
        raise ValueError("commands list is empty")
    cleaned = []
    for c in commands:
        c = (c or "").strip()
        if not c:
            raise ValueError("empty command")
        cid, sep, payload = c.partition("=")
        cid = cid.strip()
        if not cid:
            raise ValueError(f"bad command {c!r}: expected id={{json}}")
        if not any(cid.startswith(p) for p in EDIT_ALLOWLIST_PREFIXES):
            raise ValueError(f"command {cid!r} is not allowed (file-system commands are handled by this service)")
        # Parse the JSON instead of scanning the raw string: a raw scan is fooled
        # by escapes ({"path": …} → the engine still sees "path").
        if sep:
            try:
                params = json.loads(payload)
            except ValueError:
                raise ValueError(f"command {cid!r}: parameters are not valid JSON: {payload[:120]!r}")
            if not isinstance(params, dict):
                raise ValueError(f"command {cid!r}: parameters must be a JSON object")
        else:
            params = {}  # engines accept a bare id for commands without parameters
        bad = _fs_param_key(params)
        if bad:
            raise ValueError(f"command {cid!r}: parameter {bad!r} names a file — not allowed; "
                             "document files are downloaded and saved by this service")
        cleaned.append(f"{cid}={json.dumps(params, ensure_ascii=False)}")
    if len(cleaned) > 200:
        raise ValueError("too many commands in one call (max 200)")
    with tempfile.TemporaryDirectory(prefix="omd-craft-") as wd:
        src = _tmpfile(wd, source_name or "in.docx")
        out = _tmpfile(wd, out_name or ("out" + (os.path.splitext(source_name)[1] or ".docx")))
        with open(src, "wb") as f:
            f.write(data)
        # The run subcommand's flags differ per app: wordcraft uses --file/--save;
        # gridcraft and deckcraft use --in/--save (gridcraft writes with --out).
        doc_flag = "--file" if kind == "doc" else "--in"
        save_flag = "--out" if kind == "sheet" else "--save"
        args = ["run", doc_flag, src]
        for c in cleaned:
            args += ["--cmd", c]
        if kind == "doc":
            # Only WordCraft names the author of tracked changes on the CLI.
            args += ["--author", author]
        args += [save_flag, out]
        r = _run(cli, args, wd, timeout=min(MAX_EDIT_TIMEOUT, max(DEFAULT_TIMEOUT, 30 * len(cleaned))))
        if r.returncode != 0:
            raise RuntimeError(f"craft edit failed: {r.stderr.strip()[:800]}")
        with open(out, "rb") as f:
            result = f.read()
    ext = os.path.splitext(out)[1].lower()
    if ext in (".docx", ".xlsx", ".pptx"):
        _assert_zip(result, out_name or source_name)
    return result


def convert_document(kind: str, data: bytes, source_name: str, out_name: str) -> bytes:
    """Convert between formats with the doc CLI: docx→pdf, md→docx, docx→odt…"""
    _check(data)
    cli = _cli_path(kind)
    if not cli:
        raise RuntimeError(f"{kind} CLI is not available on this node")
    with tempfile.TemporaryDirectory(prefix="omd-craft-") as wd:
        src = _tmpfile(wd, source_name or "in.docx")
        out = _tmpfile(wd, os.path.basename(out_name) or "out.pdf")
        with open(src, "wb") as f:
            f.write(data)
        r = _run(cli, ["convert", src, out], wd)
        if r.returncode != 0:
            raise RuntimeError(f"craft convert failed: {r.stderr.strip()[:800]}")
        with open(out, "rb") as f:
            result = f.read()
    if os.path.splitext(out)[1].lower() in (".docx", ".xlsx", ".pptx"):
        _assert_zip(result, out_name)
    return result
