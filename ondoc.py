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
        self.header_text = ""
        self.footer_text = ""
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

    def add_mark(self, para_style=None):
        self.chars.append("\r")
        self.paras.append({"startIndex": self._st() - 1, "paragraphStyle": para_style or {}})

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
        snap["settings"] = {}
        if getattr(self, "header_text", "") or getattr(self, "footer_text", ""):
            snap["settings"]["omdHeader"] = self.header_text
            snap["settings"]["omdFooter"] = self.footer_text
        if self.images:
            snap["notes"] = {"omdImages": self.images}
        return snap


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
TEXT_NS = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
# --- odt styles registry (module scope, reset per doc) ----------------------
_OSTYLES = {}
_PARENTS = {}


def _load_od_styles(xml_text):
    """Merge one XML source into the registry (called per document part)."""
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return
    for st in root.iter():
        if _tg(st) != "style":
            continue
        name = _at(st, "name")
        if not name:
            continue
        props = {}
        for child in st:
            ct = _tg(child)
            if ct == "text-properties":
                if _at(child, "font-weight") == "bold":
                    props["bl"] = 1
                if _at(child, "font-style") == "italic":
                    props["it"] = 1
                col = _at(child, "color")
                if col and col not in ("none",):
                    props["cl"] = col
                strike = _at(child, "font-strike")
                if strike in ("single", "true", "1"):
                    props["st"] = {"s": 1}
                uline = _at(child, "text-underline-style")
                if uline and uline != "none":
                    props["ul"] = {"s": 1}
            elif ct == "paragraph-properties":
                al = _at(child, "text-align")
                if al == "center":
                    props["align"] = 2
                elif al == "right":
                    props["align"] = 3
                elif al == "justify":
                    props["align"] = 4
                elif al == "left":
                    props["align"] = 1
            elif ct == "graphic-properties":
                wrap = _at(child, "wrap")
                hpos = _at(child, "horizontal-pos")
                if wrap:
                    props["wrap"] = wrap
                if hpos:
                    props["hpos"] = hpos
        _OSTYLES[name] = props
        p_name = _at(st, "parent-style-name")
        if p_name:
            _PARENTS[name] = p_name


def _resolve_style(style_name, seen=None):
    if not style_name:
        return {}
    merged = {}
    parent = _PARENTS.get(style_name)
    if parent and parent != style_name and (seen is None or len(seen) < 8):
        seen = (seen or set()) | {style_name}
        merged = _resolve_style(parent, seen)
    merged.update(_OSTYLES.get(style_name, {}))
    return merged




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
        try:
            styles_xml = z.read("styles.xml").decode("utf-8", "replace")
        except KeyError:
            styles_xml = ""
    _load_od_styles(content)
    _load_od_styles(styles_xml)
    # headers/footers from the first master page in styles.xml
    try:
        sroot = ET.fromstring(styles_xml)
    except Exception:
        sroot = None
    if sroot is not None:
        for mp in sroot.iter():
            if _tg(mp) != "master-page":
                continue
            header = ""
            footer = ""
            for k in mp:
                if _tg(k) == "header":
                    header = " ".join(x.strip() for x, _ in inline_odt(k) if x)
                elif _tg(k) == "footer":
                    footer = " ".join(x.strip() for x, _ in inline_odt(k) if x)
            if header or footer:
                b.header_text = header
                b.footer_text = footer
            break
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
            sprops = _resolve_style(_at(node, "style-name"))
            para_style = {}
            if sprops.get("align"):
                para_style["horizontalAlign"] = sprops["align"]
            if segs:
                b.add_runs(segs, para_style or None)
            else:
                b.add_mark(para_style or None)
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
                fp = parent if parent is not None else n.getparent()
                if fp is not None:
                    anchor = _at(fp, "anchor-type") or _at(n, "anchor-type") or ""
                    gprops = _resolve_style(_at(fp, "style-name"))
                    wrap = gprops.get("wrap") or _at(fp, "wrap") or ""
                    hpos = gprops.get("hpos") or _at(fp, "horizontal-pos") or ""
                    if anchor and anchor != "as-char":
                        mk += "|anchor=" + anchor
                    if wrap:
                        mk += "|wrap=" + wrap
                    if hpos:
                        mk += "|pos=" + hpos
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
            sname = _at(n, "style-name")
            sp = _resolve_style(sname)
            if sp.get("bl"): c["bl"] = 1
            if sp.get("it"): c["it"] = 1
            if sp.get("st") and "st" not in c: c["st"] = sp["st"]
            if sp.get("ul") and "ul" not in c: c["ul"] = sp["ul"]
            if sp.get("cl") and "cl" not in c: c["cl"] = {"rgb": sp["cl"]}
            lo = sname.lower()
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

        # part XML cache while the zip is open (headers/footers/rels)
        part_xml = {}
        for nm in z.namelist():
            if nm.startswith("word/header") or nm.startswith("word/footer") or nm.startswith("word/_rels/"):
                try:
                    part_xml[nm] = z.read(nm).decode("utf-8", "replace")
                except Exception:
                    pass

    RELS = {}
    try:
        for rel in ET.fromstring(part_xml.get("word/_rels/document.xml.rels", "")):
            try:
                rid = rel.get("Id")
                tgt = rel.get("Target") or ""
                if rid and ("media/" in tgt or "image" in tgt.lower() or "header" in tgt.lower() or "footer" in tgt.lower()):
                    RELS[rid] = tgt.split("/")[-1]
            except Exception:
                continue
    except Exception:
        RELS = {}

    NS_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    NS_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    NS_WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"

    def drawing_to_marker(drawing):
        """w:drawing -> [[omd-img:..|w=..|h=..|anchor=..|pos=..]] text."""
        wp = drawing.find(NS_WP + "inline")
        if wp is None:
            wp = drawing.find(NS_WP + "anchor")
        if wp is None:
            return None
        blip = wp.find(".//" + NS_A + "blip")
        if blip is None:
            return None
        rid = blip.get(NS_R + "embed") or ""
        base = RELS.get(rid)
        if not base:
            return None
        anchor = "as-char"
        if wp is not None and wp.tag.endswith("anchor"):
            anchor = "paragraph"
        ext = wp.find(NS_WP + "extent")
        wpt = hpt = None
        if ext is not None:
            try:
                wpt = round(int(ext.get("cx")) / 12700.0, 2)
                hpt = round(int(ext.get("cy")) / 12700.0, 2)
            except Exception:
                wpt = hpt = None
        mk = "[[omd-img:" + base
        if anchor != "as-char":
            mk += "|anchor=" + anchor
            ph = wp.find(NS_WP + "positionH")
            if ph is not None:
                al = ph.find(NS_WP + "align")
                val = (al.text or "").strip().lower() if al is not None else ""
                off = ph.find(NS_WP + "posOffset")
                if not val and off is not None:
                    val = (off.text or "").strip()
                if val in ("left", "right"):
                    mk += "|pos=" + val
        if wpt:
            mk += "|w=" + str(wpt)
        if hpt:
            mk += "|h=" + str(hpt)
        return mk + "]]"

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
                drawing = child.find(W + "drawing")
                if drawing is not None:
                    mk = drawing_to_marker(drawing)
                    if mk:
                        segs.append((mk, ts))
                        continue
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
        para_style = {}
        if ppr is not None:
            st = ppr.find(W + "pStyle")
            val = st.get(W + "val", "") if st is not None else ""
            m = re.match(r"[Hh]eading(\d)", val) or re.match(r"^(\d)$", val)
            lvl = int(m.group(1)) if m else 0
            if lvl == 0 and (val or "").lower() == "title":
                lvl = 1
            lsty = ppr.find(W + "numPr")
            listmark = "\u2022 " if lsty is not None else ""
            jc = ppr.find(W + "jc")
            if jc is not None:
                jval = str(jc.get(W + "val") or "").lower()
                if jval in ("center", "right", "left", "both", "justify"):
                    para_style["horizontalAlign"] = {"left": 1, "center": 2, "right": 3, "both": 4, "justify": 4}[jval]
            prpr = ppr.find(W + "rPr")
            if prpr is not None:
                default_ts = ts_of(prpr)
        else:
            listmark = ""
        segs = runs_of(p, default_ts)
        if lvl:
            fs = HEAD_SIZE.get(lvl, 16)
            segs = [(t, dict(ts, bl=1, fs=fs)) for t, ts in segs]
            stl = {"headingId": f"Heading {lvl}", "textStyle": {"bl": 1, "fs": fs}}
            if para_style.get("horizontalAlign"):
                stl["horizontalAlign"] = para_style["horizontalAlign"]
            b.add_runs(segs, stl)
        else:
            segs0 = [s for s in segs if s[0]]
            if segs0:
                b.add_runs(segs0, para_style or None, listmark)
            else:
                b.add_mark(para_style or None)

    def docx_table(node, b):
        grid = []
        for tr in node.findall(W + "tr"):
            cells = []
            for tc in tr.findall(W + "tc"):
                lines = []
                for tpc in tc.findall(W + "p"):
                    lines.append(" ".join(x.strip() for x, _ in runs_of(tpc, {}) if x))
                cells.append(" ".join(l for l in lines if l).strip())
            grid.append(cells)
        if grid:
            b.add_table(grid)

    for el in body:
        if el.tag == W + "p":
            paragraph(el)
        elif el.tag == W + "tbl":
            docx_table(el, b)

    # header/footer parts (word/headerN.xml etc) via sectPr refs
    sroot = None
    try:
        sroot = ET.fromstring(content)
    except Exception:
        sroot = None
    if sroot is not None:
        sect = None
        for sx in sroot.iter(W + "sectPr"):
            sect = sx
            break

        def _part_text(rid):
            name = RELS.get(rid)
            if not name:
                return ""
            try:
                pxml = part_xml.get("word/" + name, "")
            except Exception:
                pxml = ""
            if not pxml:
                return ""
            try:
                proot = ET.fromstring(pxml)
            except Exception:
                return ""
            prels = {}
            try:
                relxml = part_xml.get("word/_rels/" + name + ".rels", "")
                for rel in ET.fromstring(relxml):
                    try:
                        prels[rel.get("Id")] = (rel.get("Target") or "").split("/")[-1]
                    except Exception:
                        pass
            except Exception:
                prels = {}
            A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
            WP2 = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
            R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
            mark_text = ""
            for tmp in proot.iter():
                tmp_tg = tmp.tag.split('}')[-1]
                if tmp_tg == "t":
                    if tmp.text:
                        mark_text_pieces = None
                else:
                    mark_text_pieces = None
            out = []
            for el in proot.iter():
                tg2 = el.tag.split('}')[-1]
                if tg2 == "t":
                    if el.text:
                        out.append(el.text)
                elif tg2 == "drawing":
                    wp = el.find(WP2 + "inline") or el.find(WP2 + "anchor")
                    blip = el.find(".//" + A_NS + "blip")
                    if blip is not None:
                        img_base = prels.get(blip.get(R_NS + "embed") or "")
                        if img_base:
                            ext = wp.find(WP2 + "extent") if wp is not None else None
                            mk = "[[omd-img:" + img_base
                            if ext is not None:
                                try:
                                    mk += "|w=" + str(round(int(ext.get("cx")) / 12700.0, 2))
                                    mk += "|h=" + str(round(int(ext.get("cy")) / 12700.0, 2))
                                except Exception:
                                    pass
                            mk += "]]"
                            out.append(mk)
            return " ".join(x.strip() for x in out if x.strip()).strip()

        if sect is not None:
            for ref in sect:
                tg3 = ref.tag.split('}')[-1]
                rid = ref.get(NS_R + "id")
                if not rid:
                    continue
                if tg3 == "headerReference":
                    b.header_text = b.header_text or _part_text(rid)
                elif tg3 == "footerReference":
                    b.footer_text = b.footer_text or _part_text(rid)
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


import json as _json  # noqa: E402
from datetime import datetime, date, time  # noqa: E402


# ---------------------------------------------------------------- XLSX


def xlsx_to_snapshot(path, max_rows=500, max_cols=60):
    """XLSX -> Univer sheets snapshot v1: cell values (datetimes as ISO),
    bold/italic/underline/strike + alignment styling, merged ranges."""
    import openpyxl  # lazy: a missing dependency does not crash the service
    import json as _json
    wb = openpyxl.load_workbook(path, data_only=True)
    styles = {}
    sheets = {}

    def ensure_style(ts):
        key = _json.dumps(ts, sort_keys=True)
        if key in _BY_KEY:
            return _STYLES_KEY.get(key)
        pass

    # small helpers via local structures
    style_map = {}
    style_ids = {}

    def _ensure_style_key(ts):
        key = _json.dumps(ts, sort_keys=True)
        if key not in style_ids:
            sid = 'st' + str(len(style_map) + 1)
            style_ids[key] = sid
            style_map[sid] = ts
        return style_ids[key]

    sheets_out = {}
    order = []
    for ws_idx, ws in enumerate(wb.worksheets):
        if ws_idx >= 8:
            break
        sid = 'sheet-' + str(ws_idx + 1)
        row_data = {}
        n_rows = 0
        n_cols = 0
        for r_idx, row in enumerate(ws.iter_rows(min_row=1, max_row=min(ws.max_row or 0, max_rows), max_col=min(ws.max_column or 0, max_cols))):
            cell_data = {}
            for c_idx, cell in enumerate(row):
                v = cell.value
                if v is None or v == "":
                    continue
                if isinstance(v, (datetime, date, time)):
                    v = v.isoformat(' ', 'seconds') if isinstance(v, datetime) else v.isoformat()
                ts = {}
                font = getattr(cell, 'font', None)
                if font is not None:
                    if font.bold:
                        ts['bl'] = 1
                    if font.italic:
                        ts['it'] = 1
                    if font.underline and getattr(font.underline, 'underline', None):
                        ts['ul'] = {'s': 1}
                    if font.strike is True:
                        ts['st'] = {'s': 1}
                align = getattr(cell, 'alignment', None)
                if align is not None and getattr(align, 'horizontal', None):
                    ha_val = str(align.horizontal).lower()
                    if ha_val in ('left', 'center', 'right', 'justify'):
                        ts['ha'] = {'left': 1, 'center': 2, 'right': 3, 'justify': 4}[ha_val]
                sid_style = None
                if ts:
                    sid_style = _ensure_style_key(ts)
                cell_data[str(c_idx)] = {'v': v}
                if sid_style:
                    cell_data[str(c_idx)]['s'] = sid_style
                n_cols = max(n_cols, c_idx + 1)
                n_rows = r_idx + 1
            if cell_data:
                row_data[str(r_idx)] = cell_data

        merged = []
        for rng in (ws.merged_cells.ranges or []):
            merged.append([rng.min_row - 1, rng.min_col - 1, rng.max_row - 1, rng.max_col - 1])
        sheets_out[str(sid)] = {
            'id': sid,
            'name': ws.title or ('Sheet' + str(ws_idx + 1)),
            'columnCount': (n_cols or 12) + 6,
            'rowCount': (n_rows or 30) + 8,
            'defaultRowHeight': 19.2,
            'defaultColumnWidth': 72,
            'cellData': row_data,
            'rowData': {},
            'columnData': {},
            'status': 0,
            'zoomRatio': 1,
            'merges': merged,
            'overflow': False,
            'rightToLeft': 0,
            'rowHeader': {'width': 46},
            'rowCount_i': n_rows,
            'columnIndexCount': n_cols
        }
        order.append(sid)

    return {
        'id': 'omd-sheet-' + (Path(path).stem or 'wb')[:16],
        'title': Path(path).stem or 'Workbook',
        'kind': 'sheet',
        'sheetOrder': order,
        'sheetCount': len(sheets_out),
        'styles': style_map,
        'sheets': sheets_out,
        'locale': 'en-US'
    }


def convert_xlsx_snapshot(path):
    return xlsx_to_snapshot(path)


def convert_path(path: str) -> dict:
    real = _resolve_local(path)
    if not real:
        raise HTTPException(status_code=404, detail=f"file not found on node: {path}")
    filename = os.path.basename(real)
    ext = Path(real).suffix.lower()
    if ext == ".xlsx":
        snap = xlsx_to_snapshot(real)
        snap['settings'] = {'omdKind': 'sheet'}
        return snap
    b = Builder(filename)
    if ext == ".odt":
        parse_odt(real, b)
    elif ext == ".docx":
        parse_docx(real, b)
    else:
        raise HTTPException(status_code=415, detail=f"unsupported format: {ext} (odt, docx, xlsx)")
    return b.snapshot()
