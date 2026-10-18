"""
Standalone tests for craft_service (run: venv/bin/python tests/test_craft_service.py).

Covers the guards added in the WordCraft MCP integration review:
- local-disk shortcut is owner/private only and stays inside the session storage,
- edit_document parses command JSON and rejects filesystem-path parameters
  (including the {"path": ...} unicode-escape bypass),
- create_document honours the target extension (.csv must not become XLSX),
- the deckcraft command catalog is readable (no --json there),
- local saves are atomic and create missing parent folders.
"""

import json
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import craft_service as cs


class _Ctx:
    """Minimal UserContext stand-in."""

    def __init__(self, private_mode=False, omd_key="", storage="", trusted_owner=False):
        self.private_mode = private_mode
        self.omd_key = omd_key
        self.storage = storage
        self.trusted_owner = trusted_owner


def expect_error(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), f"expected {needle!r} in error, got: {e}"
        return
    raise AssertionError(f"expected ValueError containing {needle!r}")


def test_edit_allowlist():
    # file-system command ids stay blocked
    expect_error(lambda: cs.edit_document("doc", b"x", "a.docx", ['file.open={"path":"/etc/passwd"}']),
                 "not allowed")
    # review.compare reads `original` from disk — blocked (string value)
    expect_error(lambda: cs.edit_document("doc", b"x", "a.docx",
                                          ['review.compare={"text":"x","original":"/tmp/s.docx"}']),
                 "'original'")
    # JSON unicode-escape bypass: {"path": ...} — the engine decodes it, so we parse too
    cmd = 'data.fromTextCsv={"\\u0070ath": "/tmp/x.csv"}'
    assert '"path"' not in cmd, "test premise: raw scan must not see the key"
    expect_error(lambda: cs.edit_document("sheet", b"x", "a.xlsx", [cmd]), "'path'")
    # invalid JSON payload
    expect_error(lambda: cs.edit_document("doc", b"x", "a.docx", ["text.insert=hello"]), "not valid JSON")
    # not a JSON object
    expect_error(lambda: cs.edit_document("doc", b"x", "a.docx", ["text.insert=[1]"]), "JSON object")


def test_fs_param_key():
    assert cs._fs_param_key({"path": "/x"}) == "path"
    assert cs._fs_param_key({"text": "x", "original": "/y"}) == "original"
    assert cs._fs_param_key({"start": {"path": ["a"], "off": 1}}) is None  # geometry list
    assert cs._fs_param_key({"path": [0]}) is None                          # story path list
    assert cs._fs_param_key({"data": "AAAA", "name": "f.ttf"}) is None
    assert cs._fs_param_key({"text": "clean"}) is None


def test_allowed_command_actually_runs():
    # a normal allowlisted command must still go through and edit the document
    data = cs.create_document("doc", "t.docx", "# Title\n\nBody text.")
    out = cs.edit_document("doc", data, "t.docx", ["document.paragraph={\"path\":[0]}", "caret.docEnd={}"])
    assert zipfile.is_zipfile(__import__("io").BytesIO(out))
    assert "Body text." in cs.read_document_text("doc", out, "t.docx")


def test_create_document_extensions():
    csv_bytes = cs.create_document("sheet", "report.csv", "a,b\n1,2")
    assert not zipfile.is_zipfile(__import__("io").BytesIO(csv_bytes)), ".csv target must not be XLSX"
    assert csv_bytes.decode("utf-8", "replace").startswith("a,b")
    xlsx_bytes = cs.create_document("sheet", "report.xlsx", "a,b\n1,2")
    assert zipfile.is_zipfile(__import__("io").BytesIO(xlsx_bytes))


def test_resolve_local_path_gate():
    old_node, old_root = cs.node_name, cs.storage_root
    tmp = tempfile.mkdtemp(prefix="craft-root-")
    os.makedirs(os.path.join(tmp, "Private"))
    doc = os.path.join(tmp, "Private", "note.docx")
    with open(doc, "wb") as f:
        f.write(b"x")
    cs.node_name, cs.storage_root = lambda: "testnode", lambda: tmp
    env_old = os.environ.pop("OMD_CRAFT_ALLOW_LOCAL", None)
    try:
        # a client-supplied key without private_mode never opens the disk (the hole)
        guest = _Ctx(private_mode=False, omd_key="anything", storage="testnode")
        assert cs.resolve_local_path(guest, "/testnode/Private/note.docx") is None
        # pm=1 shared guest: private, but confined to its declared storage subtree
        shared = _Ctx(private_mode=True, omd_key="k", storage="testnode/Private")
        assert cs.resolve_local_path(shared, "/testnode/Private/note.docx") == doc
        assert cs.resolve_local_path(shared, "/testnode/other/file.txt") is None
        # NODE OWNER (tunnel role=owner → api sets trusted_owner): any path of
        # their own node inside STORAGE_ROOT — no pointless trip through the gateway
        os.makedirs(os.path.join(tmp, "other"))
        with open(os.path.join(tmp, "other", "file.txt"), "wb") as f:
            f.write(b"y")
        owner = _Ctx(private_mode=True, omd_key="k", storage="testnode/Private", trusted_owner=True)
        assert cs.resolve_local_path(owner, "/testnode/Private/note.docx") == doc
        assert cs.resolve_local_path(owner, "/testnode/other/file.txt") == os.path.join(tmp, "other", "file.txt")
        # other node's name → gateway even for the owner
        assert cs.resolve_local_path(owner, "/othernode/Private/note.docx") is None
        # traversal out of STORAGE_ROOT
        assert cs.resolve_local_path(owner, "/testnode/../etc/passwd") is None
        # kill switch
        os.environ["OMD_CRAFT_ALLOW_LOCAL"] = "0"
        assert cs.resolve_local_path(owner, "/testnode/Private/note.docx") is None
    finally:
        cs.node_name, cs.storage_root = old_node, old_root
        if env_old is not None:
            os.environ["OMD_CRAFT_ALLOW_LOCAL"] = env_old
        else:
            os.environ.pop("OMD_CRAFT_ALLOW_LOCAL", None)


def test_deckcraft_catalog():
    cli = cs._cli_path("deck")
    if not cli:
        print("  (deckcraft-cli not installed — skipped)")
        return
    from mcp_server import _engine_command_ids
    ids = _engine_command_ids(cli)
    assert ids, "deckcraft command catalog must not be empty"
    assert any(i.startswith("slide.") for i in ids), ids[:10]


def test_atomic_local_write():
    from mcp_server import _write_local_atomic
    tmp = tempfile.mkdtemp(prefix="craft-atomic-")
    target = os.path.join(tmp, "new", "sub", "doc.docx")
    msg = _write_local_atomic(target, "/node/new/sub/doc.docx", b"data")
    assert os.path.isfile(target) and open(target, "rb").read() == b"data"
    assert "4 bytes" in msg
    assert not [n for n in os.listdir(os.path.dirname(target)) if n.endswith(".omd-tmp")]
    # overwrite is atomic too
    _write_local_atomic(target, "/node/new/sub/doc.docx", b"more")
    assert open(target, "rb").read() == b"more"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"OK   {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
