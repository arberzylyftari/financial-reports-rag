"""Parse HTML financial report files into clean text with metadata."""

import re
from pathlib import Path
from bs4 import BeautifulSoup, Tag


def _table_to_text(table: Tag) -> str:
    """Convert an HTML table into readable labeled sentences.

    Each data row is rendered as:
        "Label: col_header_1 value1, col_header_2 value2"

    This preserves the relationship between row labels (e.g. "R&D Expense")
    and their values across columns (e.g. years), which plain get_text() destroys.
    """
    rows = table.find_all("tr")
    if not rows:
        return ""

    # Extract column headers from the first row whose cells are <th> tags.
    headers: list[str] = []
    for row in rows:
        ths = row.find_all("th")
        if ths:
            headers = [th.get_text(" ", strip=True) for th in ths]
            break

    lines: list[str] = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        texts = [c.get_text(" ", strip=True) for c in cells]
        # Skip rows that are all empty or duplicate the header row.
        if not any(texts) or texts == headers:
            continue

        if headers and len(texts) > 1:
            label = texts[0]
            # Pair each value cell with its column header when available.
            pairs: list[str] = []
            for i, val in enumerate(texts[1:], 1):
                if not val:
                    continue
                col_header = headers[i] if i < len(headers) else ""
                pairs.append(f"{col_header} {val}".strip() if col_header else val)
            if pairs:
                lines.append(f"{label}: {', '.join(pairs)}")
        else:
            # No headers — just join non-empty cells with a pipe separator.
            lines.append(" | ".join(t for t in texts if t))

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
