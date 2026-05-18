"""Parse HTML financial report files into clean text with metadata."""

import re
from pathlib import Path
from bs4 import BeautifulSoup, Tag


_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_NUM_RE = re.compile(r"^\(?-?\$?[\d,]+(?:\.\d+)?\)?%?$")


def _merge_paren_negatives(cells: list[str]) -> list[str]:
    """Merge a negative number split as '(' '1,234' ')' back into '(1,234)'."""
    out: list[str] = []
    i = 0
    while i < len(cells):
        if cells[i] == "(" and i + 2 < len(cells) and cells[i + 2] == ")":
            out.append(f"({cells[i + 1]})")
            i += 3
        else:
            out.append(cells[i])
            i += 1
    return out


def _is_number(s: str) -> bool:
    """True if the cell is a financial value (handles $, commas, %, parens)."""
    return bool(_NUM_RE.match(s.replace(" ", "")))


def _table_to_text(table: Tag) -> str:
    """Convert an HTML table into readable labeled sentences.

    Financial-statement rows become:
        "Research and development: 2024 31,370, 2023 29,915, 2022 26,251"

    preserving the link between a row label and each fiscal year's value,
    which plain get_text() destroys. SEC tables split "$", values and "%"
    into separate <td> cells and rarely use <th> for year headers, so the
    year-header row is detected by content and noise cells are dropped before
    pairing. Tables with no detectable year row (non-financial) fall back to
    a simple readable join.
    """
    rows = table.find_all("tr")
    if not rows:
        return ""

    grid: list[list[str]] = []
    for row in rows:
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        cells = _merge_paren_negatives(cells)
        cleaned = [c for c in cells if c not in ("", "$", "%")]
        if cleaned:
            grid.append(cleaned)

    # Locate the year-header row: first row with >=2 cells containing a year.
    header_years: list[str] = []
    header_idx = -1
    for idx, cells in enumerate(grid):
        year_cells = [m.group() for c in cells if (m := _YEAR_RE.search(c))]
        if len(year_cells) >= 2:
            header_years = year_cells
            header_idx = idx
            break

    if header_idx == -1:
        # No year header — non-financial table; keep a readable flat join.
        return "\n".join(" | ".join(cells) for cells in grid)

    lines: list[str] = []
    for idx, cells in enumerate(grid):
        if idx == header_idx:
            continue
        first_num = next(
            (i for i, c in enumerate(cells) if _is_number(c)), len(cells)
        )
        label = " ".join(cells[:first_num]).strip(" :")
        values = [c for c in cells[first_num:] if _is_number(c)]

        if not label and not values:
            continue
        if not values:
            lines.append(label)
        elif len(values) <= len(header_years):
            pairs = [f"{header_years[i]} {v}" for i, v in enumerate(values)]
            lines.append(f"{label}: {', '.join(pairs)}" if label else ", ".join(pairs))
        else:
            # More values than year columns (extra %-of-sales subcolumns):
            # don't assert year mappings — list values plainly.
            lines.append(f"{label}: {', '.join(values)}" if label else ", ".join(values))

    return "\n".join(lines)


def _extract_year_from_filename(filename: str) -> int | None:
    """Extract the fiscal year from a filename.

    Handles SEC-style names like aapl-20240928.htm (8-digit date) and
    arbitrary names like tm252787d2_10ka.htm by looking for a 4-digit
    sequence in the range 2000-2099.
    """
    # Prefer an 8-digit date block: first 4 digits are the year.
    match = re.search(r"(20\d{2})\d{4}", filename)
    if match:
        return int(match.group(1))
    # Fall back to any standalone 4-digit year in range 2000-2099.
    match = re.search(r"\b(20\d{2})\b", filename)
    if match:
        return int(match.group(1))
    return None


def _clean_text(text: str) -> str:
    """Normalize whitespace, remove empty lines, and replace non-ASCII symbols."""
    # Replace common non-ASCII bullet/arrow characters with plain ASCII equivalents
    replacements = {
        "•": "-",  "‣": "-",  "●": "-",  "○": "-",
        "◉": "-",  "◦": "-",  "⦿": "-",  "⁃": "-",
        "·": "-",  "∙": "-",  "▪": "-",  "▸": "-",
        "–": "-",  "—": "-",
        "‘": "'",  "’": "'",
        "“": '"',  "”": '"',
        " ": " ",
        "�": "",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    lines = (line.strip() for line in text.splitlines())
    non_empty = (line for line in lines if line)
    return "\n".join(non_empty)


def _extract_year_from_text(text: str) -> int | None:
    """Extract fiscal year from document text by looking for 'December 31, 20XX' or 'fiscal 20XX'."""
    # 10-K filings typically state the period end date near the top
    for pattern in [
        r"(?:December\s+31,?\s+)(20\d{2})",
        r"(?:fiscal\s+year\s+ended?\s+\S+\s+\d{1,2},?\s+)(20\d{2})",
        r"(?:year\s+ended?\s+\S+\s+\d{1,2},?\s+)(20\d{2})",
        r"(?:Annual\s+Report.*?)(20\d{2})",
    ]:
        match = re.search(pattern, text[:5000], re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def parse_file(filepath: Path, company: str) -> dict | None:
    """Parse a single HTML financial report file.

    Returns a dict with keys 'text' and 'metadata', or None on failure.
    """
    try:
        try:
            raw = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = filepath.read_text(encoding="cp1252")
        soup = BeautifulSoup(raw, "lxml")

        for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            tag.decompose()

        # Replace every <table> in-place with structured plain text so that
        # financial figures keep their row labels and column year headers.
        for table in soup.find_all("table"):
            readable = _table_to_text(table)
            table.replace_with(readable + "\n")

        text = _clean_text(soup.get_text(separator="\n"))

        year = _extract_year_from_filename(filepath.name)
        if year is None:
            year = _extract_year_from_text(text)

        return {
            "text": text,
            "metadata": {
                "company": company,
                "year": year,
                "source_file": filepath.name,
            },
        }
    except Exception as exc:
        print(f"[parser] ERROR: failed to parse {filepath.name}: {exc}")
        return None


def parse_directory(data_dir: Path) -> list[dict]:
    """Parse all HTML files from every subdirectory under data/raw/.

    Each subdirectory name becomes the company name (e.g. data/raw/nvidia → Nvidia).
    Returns a list of document dicts: {text, metadata}.
    """
    documents = []
    raw_dir = data_dir / "raw"
    company_dirs = {
        d.name.capitalize(): d
        for d in sorted(raw_dir.iterdir())
        if d.is_dir()
    }

    for company, company_path in company_dirs.items():
        if not company_path.exists():
            print(f"[parser] WARNING: directory not found: {company_path}")
            continue

        html_files = list(company_path.glob("*.htm")) + list(company_path.glob("*.html"))
        if not html_files:
            print(f"[parser] WARNING: no HTML files found in {company_path}")
            continue

        for filepath in sorted(html_files):
            print(f"[parser] Parsing {company} — {filepath.name}")
            doc = parse_file(filepath, company)
            if doc:
                documents.append(doc)
                print(f"[parser]   → {len(doc['text']):,} chars extracted (year={doc['metadata']['year']})")

    print(f"[parser] Done. {len(documents)} document(s) parsed.")
    return documents
