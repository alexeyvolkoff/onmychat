"""On My Chat — native document converter (.odt / .docx -> Univer docs snapshot).

Reads the ORIGINAL file from the node's LOCAL filesystem (never via the
gateway), converts the richest supported representation into the Univer
IDocumentData snapshot ("univer.json") and returns it.

Endpoints (mount into api.py):
    ondoc.mount(app)

    GET  /ondoc/health
    POST /ondoc/convert   body: {path: "/Documents/x.odt"} -> {snapshot: {...}, images: n}

The client then renders and edits the snapshot directly and persists it as
<name>.univer.json on the drive (import phase; write-back to the original
format is a later phase).
"""

import base64
import logging
import os
import re
import uuid
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

log = logging.getLogger("ondoc")

PAGE = {"width": 595.28, "height": 841.89}
MARGIN = 72
HEAD_SIZE = {1: 24, 2: 20, 3: 16, 4: 14, 5: 13, 6: 12}


def _len_to_pt(raw):
    """Convert odt length (in/cm/mm/pt/px/pc) to pt float; None when unknown."""
    if not raw:
        return None
    m = re.match(r"([0-9.]+)\s*(in|cm|mm|pt|px|pc)?", raw.strip())
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2) or "pt"
    factor = {"in": 72.0, "cm": 72.0 / 2.54, "mm": 72.0 / 25.4, "pt": 1.0, "px": 0.75, "pc": 12.0}.get(unit, 1.0)
    return round(val * factor, 2)


def _tg(node):
    return node.tag.split('}')[-1]


def _at(node, key):
    for k, v in node.attrib.items():
        if k.split('}')[-1] == key:
            return v
    return ''


class Builder:
    def __init__(self, title):
        self.title = title
        self.chars = []
        self.paras = []
        self.runs = []      # [st, ed, ts]
        self.images = []
        self.tables = {}    # tableId -> ITable

    def _st(self):
        return len("".join(self.chars))

    def add_runs(self, segs, para_style=None, listmark=""):
        for i, (text, ts) in enumerate(segs):
            text2 = (listmark + text) if (listmark and i == 0) else text
            if not text2:
                continue
            st = self._st()
            self.chars.append(text2)
            ed = self._st()
            if ts:
                self.runs.append([st, ed, dict(ts)])
        self.chars.append("\r")
        self.paras.append({"startIndex": self._st() - 1, "paragraphStyle": para_style or {}})

    def add_mark(self):
        self.chars.append("\r")
        self.paras.append({"startIndex": self._st() - 1, "paragraphStyle": {}})

    def add_table(self, rows):
        """Emit a native docs table: DataStreamTreeTokenType markers plus an
        ITable entry (cells carry inline \r paragraph + \n section break,
        mirroring the docs-ui genEmptyTable shape)."""
        rows = [r for r in rows if r]
        if not rows:
            return
        cols = max(len(r) for r in rows)
        for r in rows:
            while len(r) < cols:
                r.append("")
        ST, RS, CS, CE, RE, TE = "\x1A", "\x1B", "\x1C", "\x1D", "\x0E", "\x0F"
        tid = "tbl-" + uuid.uuid4().hex[:8]
        table_rows = []
        self.chars.append(ST)
        for row in rows:
            self.chars.append(RS)
            cells = []
            for cell in row:
                self.chars.append(CS)
                self.chars.append(cell)
                self.chars.append("\r")
                self.paras.append({"startIndex": self._st() - 1, "paragraphStyle": {}})
                self.chars.append("\n")
                self.chars.append(CE)
                cells.append({})
            self.chars.append(RE)
            table_rows.append({"tableCells": cells, "trHeight": {"val": {"v": 22}, "hRule": 0}})
        self.chars.append(TE)
        content_w = PAGE["width"] - 2 * MARGIN
        self.tables[tid] = {
            "tableId": tid,
            "tableRows": table_rows,
            "tableColumns": [{"size": {"type": 1, "width": {"v": int(content_w / cols / 0.75)}}} for _ in range(cols)],
            "align": 0, "indent": {"v": 0}, "textWrap": 0,
            "position": {
                "positionH": {"relativeFrom": 0, "posOffset": 0},
                "positionV": {"relativeFrom": 0, "posOffset": 0},
            },
            "dist": {"distB": 0, "distL": 0, "distR": 0, "distT": 0},
            "size": {"type": 0, "width": {"v": int(content_w / 0.75)}},
            "cellMargin": {"start": {"v": 8}, "end": {"v": 8}, "top": {"v": 4}, "bottom": {"v": 4}},
        }

    def snapshot(self):
        data = "".join(self.chars) or "\r\n"
        if not data.endswith("\r\n"):
            data = data.rstrip("\r") + "\r\n"
        merged = []
        for st, ed, ts in self.runs:
            if st >= ed:
                continue
            if merged and merged[-1][1] == st and merged[-1][2] == ts:
                merged[-1][1] = ed
                continue
            merged.append([st, ed, ts])
        body = {
            "dataStream": data,
            "textRuns": [{"st": s, "ed": e, "ts": dict(t)} for s, e, t in merged],
            "paragraphs": self.paras,
            "sectionBreaks": [{"startIndex": len(data) - 1}],
        }
        if self.tables:
            body["tables"] = dict(self.tables)
        snap = {
            "id": "omd-doc-" + uuid.uuid4().hex[:12],
            "title": self.title,
            "body": body,
            "documentStyle": {
                "pageSize": PAGE,
                "marginTop": MARGIN, "marginBottom": MARGIN,
                "marginLeft": MARGIN, "marginRight": MARGIN,
            },
            "settings": {},
        }
        if self.images:
            snap["notes"] = {"omdImages": self.images}
        return snap


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
TEXT_NS = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"


def collect_images(zdir, b, prefix):
    for name in zdir.namelist():
        if not name.startswith(prefix):
            continue
        ext = Path(name).suffix.lower().lstrip('.')
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "bmp": "image/bmp", "webp": "image/webp"}.get(ext)
        if not mime:
            continue
        try:
            b.images.append({
                "name": Path(name).name,
                "mime": mime,
                "data": base64.b64encode(zdir.read(name)).decode("ascii"),
            })
        except Exception:
            log.exception("image read failed: %s", name)


# ---------------------------------------------------------------------- ODT

def parse_odt(path, b):
    with ZipFile(path) as z:
        content = z.read("content.xml").decode("utf-8", "replace")
        collect_images(z, b, "Pictures/")
    root = ET.fromstring(content)
    textroot = None
    for el in root:
        if _tg(el) == "body":
            for c in el:
                if _tg(c) == "text":
                    textroot = c
            break
    if textroot is None:
        raise ValueError("no office:text in odt")

    def block(node):
        tg = _tg(node)
        if tg == "p":
            segs = [s for s in inline_odt(node) if s[0]]
            if segs:
                b.add_runs(segs)
            else:
                b.add_mark()
        elif tg == "h":
            od_heading(node, b)
        elif tg == "list":
            od_list(node, b)
        elif tg == "table":
            od_table(node, b)

    for el in textroot:
        if _tg(el) in ("p", "h", "list", "table"):
            block(el)
    return b


def inline_odt(node, ts=None):
    out = []
    ts = dict(ts or {})
    def walk(n, cur, parent=None):
        t = _tg(n)
        if t == "s":
            out.append((" " * max(1, int(_at(n, "c") or 1)), dict(cur)))
        elif t == "tab":
            out.append(("\t", dict(cur)))
        elif t == "line-break":
            out.append((" ", dict(cur)))
        elif t == "image":
            href = _at(n, "href") or _at(n, "*href") or ""
            base = href.split("/")[-1].split("#")[-1].strip()
            if base:
                mk = "[[omd-img:" + base
                fp = parent
                if fp is not None:
                    wpt = None
                    hpt = None
                    for akey, aval in fp.attrib.items():
                        tail = akey.split('}')[-1]
                        if tail == "width" and wpt is None:
                            wpt = _len_to_pt(aval)
                        elif tail == "height" and hpt is None:
                            hpt = _len_to_pt(aval)
                    if wpt:
                        mk += "|w=" + str(wpt)
                    if hpt:
                        mk += "|h=" + str(hpt)
                out.append((mk + "]]", dict(cur)))
                if n.tail:
                    out.append((n.tail, dict(cur)))
                return
        else:
            c = dict(cur)
            lo = _at(n, "style-name").lower()
            if "bold" in lo or "strong" in lo:
                c["bl"] = 1
            if "italic" in lo:
                c["it"] = 1
            if n.text:
                out.append((n.text, dict(c)))
            for ch in n:
                        walk(ch, c, n)
        if n.tail:
            out.append((n.tail, dict(cur)))
    if node.text:
        out.append((node.text, dict(ts)))
    for ch in node:
        walk(ch, ts)
    return out


def od_heading(node, b):
    lvl = int(_at(node, "outline-level") or "1") or 1
    fs = HEAD_SIZE.get(lvl, 16)
    segs = [(t, dict(ts, bl=1, fs=fs)) for t, ts in inline_odt(node) if t]
    b.add_runs(segs, {"headingId": f"Heading {lvl}", "textStyle": {"bl": 1, "fs": fs}})


def od_list(node, b):
    ordered = "num" in (_at(node, "style-name") or "").lower()
    counter = 0
    for item in node:
        if _tg(item) != "list-item":
            continue
        counter += 1
        mark = f"{counter}. " if ordered and b is not None else "• "
        for c in item:
            ct = _tg(c)
            if ct == "p":
                segs = [x for x in inline_odt(c) if x[0]]
                first = True
                built = []
                for t, ts in segs:
                    built.append(((mark + t) if first else t, ts))
                    first = False
                if built:
                    b.add_runs(built, {"bullet": {"listId": "bullet", "nestingLevel": 0}})
                else:
                    b.add_mark()
            elif ct == "h":
                segs = [x for x in inline_odt(c) if x[0]]
                first = True
                built = []
                for t, ts in segs:
                    built.append(((mark + t) if first else t, ts))
                    first = False
                if built:
                    b.add_runs(built, {})
            elif ct == "list":
                od_list(c, b)
            elif ct == "table":
                od_table(c, b)


def od_table(node, b):
    grid = []
    for tr in node:
        if _tg(tr) != "table-row":
            continue
        cells = []
        for tc in tr:
            if _tg(tc) != "table-cell":
                continue
            cells.append(" ".join(t for t, _ in inline_odt(tc) if t).strip())
        grid.append(cells)
    if grid:
        b.add_table(grid)


# ---------------------------------------------------------------------- DOCX

def parse_docx(path, b):
    with ZipFile(path) as z:
        content = z.read("word/document.xml").decode("utf-8", "replace")
        collect_images(z, b, "word/media/")
    root = ET.fromstring(content)
    body = None
    for el in root:
        if el.tag == W + "body":
            body = el
            break
    if body is None:
        raise ValueError("no w:body in docx")

    def ts_of(rpr, base=None):
        ts = dict(base or {})
        if rpr is None:
            return ts

        def is_on(tag):
            e = rpr.find(W + tag)
            if e is None:
                return False
            v = e.get(W + "val")
            return v not in ("false", "0", "none")
        if is_on("b"):
            ts["bl"] = 1
        if is_on("i"):
            ts["it"] = 1
        if is_on("u"):
            ts["ul"] = {"s": 1}
        if is_on("strike"):
            ts["st"] = {"s": 1}
        sz = rpr.find(W + "sz")
        if sz is not None:
            try:
                ts["fs"] = int(sz.get(W + "val")) / 2
            except Exception:
                pass
        col = rpr.find(W + "color")
        if col is not None:
            v = col.get(W + "val")
            if v and v.lower() not in ("auto", "000000"):
                ts["cl"] = {"rgb": "#" + v}
        return ts

    def runs_of(p, default_ts):
        segs = []
        for child in p:
            tag = child.tag
            if tag == W + "r":
                rpr = child.find(W + "rPr")
                ts = dict(ts_of(rpr, default_ts))
                text = "".join(t.text or "" for t in child.findall(W + "t"))
                br = child.find(W + "br")
                if br is not None:
                    text = " " if not text else text
                if text:
                    segs.append((text, ts))
            elif tag == W + "hyperlink":
                for r in child.findall(W + "r"):
                    ts = dict(ts_of(r.find(W + "rPr"), default_ts))
                    ts.setdefault("cl", {"rgb": "#2c53f1"})
                    ts.setdefault("ul", {"s": 1})
                    text = "".join(t.text or "" for t in r.findall(W + "t"))
                    if text:
                        segs.append((text, ts))
        return segs

    def heading_level(p):
        ppr = p.find(W + "pPr")
        if ppr is None:
            return 0
        st = ppr.find(W + "pStyle")
        val = st.get(W + "val", "") if st is not None else ""
        m = re.match(r"[Hh]eading(\d)", val) or re.match(r"^(\d)$", val)
        return int(m.group(1)) if m else 0

    def paragraph(p):
        default_ts = {}
        ppr = p.find(W + "pPr")
        lvl = 0
        if ppr is not None:
            st = ppr.find(W + "pStyle")
            val = st.get(W + "val", "") if st is not None else ""
            m = re.match(r"[Hh]eading(\d)", val) or re.match(r"^(\d)$", val)
            lvl = int(m.group(1)) if m else 0
            if lvl == 0 and (val or "").lower() == "title":
                lvl = 1
            lsty = ppr.find(W + "numPr")
            listmark = "• " if lsty is not None else ""
            prpr = ppr.find(W + "rPr")
            if prpr is not None:
                default_ts = ts_of(prpr)
        else:
            listmark = ""
        segs = runs_of(p, default_ts)
        if lvl:
            fs = HEAD_SIZE.get(lvl, 16)
            segs = [(t, dict(ts, bl=1, fs=fs)) for t, ts in segs]
            b.add_runs(segs, {"headingId": f"Heading {lvl}", "textStyle": {"bl": 1, "fs": fs}})
        else:
            segs0 = [s for s in segs if s[0]]
            if segs0:
                b.add_runs(segs0, {}, listmark)
            else:
                b.add_mark()

    def docx_table(node, b):
        for tr in node.findall(W + "tr"):
            cells = []
            for tc in tr.findall(W + "tc"):
                cells.append("".join(t.text or "" for t in tc.iter(W + "t")))
            if cells:
                b.add_runs([(" | ".join(cells) + " | ", {"fs": 11})], {"textStyle": {}})

    for el in body:
        if el.tag == W + "p":
            paragraph(el)
        elif el.tag == W + "tbl":
            docx_table(el, b)
    return b


# ------------------------------------------------------------ HTTP endpoints

from fastapi import FastAPI, Request, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class ConvertBody(BaseModel):
    path: str


def _resolve_local(source: str) -> str | None:
    """Mirror of api.py /rag/import/local path resolution: device-relative
    /<share>/<file> -> node-local /home/<osuser>/<share>/<file>."""
    import getpass
    if os.path.isfile(source):
        return source
    candidates = []
    if source.startswith("/") and not source.startswith("/home/"):
        candidates.append(f"/home/{getpass.getuser()}{source}")
    candidates.append(f"/home/{getpass.getuser()}{source}")
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return None


def convert_path(path: str) -> dict:
    real = _resolve_local(path)
    if not real:
        raise HTTPException(status_code=404, detail=f"file not found on node: {path}")
    filename = os.path.basename(real)
    ext = Path(real).suffix.lower()
    b = Builder(filename)
    if ext == ".odt":
        parse_odt(real, b)
    elif ext == ".docx":
        parse_docx(real, b)
    else:
        raise HTTPException(status_code=415, detail=f"unsupported format: {ext} (odt, docx)")
    return b.snapshot()


def mount(app: FastAPI):
    @app.get("/ondoc/health")
    async def ondoc_health():
        return {"ok": True, "formats": ["odt", "docx"]}

    @app.post("/ondoc/convert")
    async def ondoc_convert(body: ConvertBody, request: Request):
        if not body.path or body.path.startswith("http"):
            raise HTTPException(status_code=422, detail="device-local path is required")
        snapshot = convert_path(body.path)
        return {"ok": True, "path": body.path, "snapshot": snapshot}
