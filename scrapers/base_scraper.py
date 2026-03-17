"""
Base scraper: shared HTTP fetching, disk caching, rate limiting, and
HTML table parsing utilities used by all scrapers in this project.

Sports-reference.com wraps many stats tables in HTML comments to defer
rendering. The `parse_sr_table` helper handles both visible and comment-
wrapped tables automatically.
"""

import hashlib
import io
import os
import re
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})

# Seconds to wait between requests (be polite to sports-reference)
REQUEST_DELAY = 3.5
_last_request_time: float = 0.0


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(url: str, cache_subdir: str) -> Path:
    """Return a deterministic file path for caching a URL's HTML."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / cache_subdir / f"{url_hash}.html"


def fetch(url: str, cache_subdir: str = "misc", force_refresh: bool = False) -> str:
    """
    Fetch a URL, returning raw HTML.

    Results are cached to disk. If a cached copy exists it is returned
    immediately without hitting the network (unless force_refresh=True).
    """
    global _last_request_time

    cache_file = _cache_path(url, cache_subdir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not force_refresh:
        logger.debug("Cache hit: %s", url)
        return cache_file.read_text(encoding="utf-8")

    # Enforce rate limit
    elapsed = time.time() - _last_request_time
    if elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)

    logger.info("Fetching: %s", url)
    response = SESSION.get(url, timeout=30)
    _last_request_time = time.time()

    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        logger.warning("Rate limited. Sleeping %ds...", retry_after)
        time.sleep(retry_after)
        response = SESSION.get(url, timeout=30)
        _last_request_time = time.time()

    response.raise_for_status()
    html = response.text
    cache_file.write_text(html, encoding="utf-8")
    return html


# ---------------------------------------------------------------------------
# HTML / table parsing
# ---------------------------------------------------------------------------

def _uncomment_tables(html: str) -> str:
    """
    Sports-reference hides some tables inside HTML comments.
    Strip the comment wrappers so BeautifulSoup can find them.
    """
    soup = BeautifulSoup(html, "lxml")
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in comment:
            new_tag = BeautifulSoup(comment, "lxml")
            comment.replace_with(new_tag)
    return str(soup)


def parse_sr_table(html: str, table_id: str) -> Optional[pd.DataFrame]:
    """
    Parse a sports-reference HTML stats table by its id attribute.

    Handles tables that are wrapped in HTML comments (common on SR pages).
    Returns a cleaned DataFrame, or None if the table is not found.

    Multi-level headers are collapsed to a single row by joining non-empty
    levels with '_'.
    """
    processed = _uncomment_tables(html)
    soup = BeautifulSoup(processed, "lxml")
    table = soup.find("table", {"id": table_id})

    if table is None:
        logger.warning("Table '%s' not found in page.", table_id)
        return None

    # pandas read_html handles colspan/rowspan in headers well
    dfs = pd.read_html(io.StringIO(str(table)), header=[0, 1])
    if not dfs:
        return None

    df = dfs[0]

    # Flatten multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        def flatten(col):
            parts = [str(c).strip() for c in col if str(c) not in ("", "nan", "Unnamed")]
            return "_".join(parts) if parts else "_".join(str(c) for c in col)
        df.columns = [flatten(c) for c in df.columns]
    else:
        df.columns = [str(c).strip() for c in df.columns]

    # Drop rows that are repeated header rows (SR inserts these every 25 rows)
    if "School" in df.columns:
        df = df[df["School"].notna() & (df["School"] != "School")]
    elif "Tm" in df.columns:
        df = df[df["Tm"].notna() & (df["Tm"] != "Tm")]

    return df.reset_index(drop=True)


def get_soup(html: str) -> BeautifulSoup:
    """Return a BeautifulSoup of the HTML with comment tables exposed."""
    return BeautifulSoup(_uncomment_tables(html), "lxml")


# ---------------------------------------------------------------------------
# Team name normalisation
# ---------------------------------------------------------------------------

# Map common SR name variants to a canonical form used across all data.
# Extend this dict as mismatches are discovered.
NAME_FIXES = {
    "Connecticut": "UConn",
    "Miami (FL)": "Miami",
    "Miami (Ohio)": "Miami OH",
    "Louisiana State": "LSU",
    "Mississippi": "Ole Miss",
    "Mississippi State": "Mississippi St.",
    "Texas Christian": "TCU",
    "Southern Methodist": "SMU",
    "Pittsburgh": "Pitt",
    "St. John's (NY)": "St. John's",
    "Virginia Commonwealth": "VCU",
    "North Carolina State": "NC State",
    "University of California": "California",
    "Brigham Young": "BYU",
    "Central Florida": "UCF",
    "Alabama-Birmingham": "UAB",
    "Massachusetts": "UMass",
    "Nevada-Las Vegas": "UNLV",
    "Texas-El Paso": "UTEP",
    "Texas-San Antonio": "UTSA",
    "Arkansas-Little Rock": "Little Rock",
    "Charleston": "College of Charleston",
    "Southern California": "USC",
    "Saint Mary's (CA)": "Saint Mary's",
    "Saint Mary's (Cal)": "Saint Mary's",
    "Long Island University": "LIU",
    "Cal State Fullerton": "CS Fullerton",
    "Cal State Bakersfield": "CS Bakersfield",
    "Cal State Northridge": "CSUN",
    # Stats page full name → bracket abbreviation
    "North Carolina": "UNC",
    "Maryland-Baltimore County": "UMBC",
    "Pennsylvania": "Penn",
    "East Tennessee State": "ETSU",
    "McNeese State": "McNeese",
    "Louisiana-Lafayette": "Louisiana",
    "Louisiana-Monroe": "UL Monroe",
    "Arkansas-Pine Bluff": "UAPB",
    "Texas A&M-Corpus Christi": "TAMUCC",
    "Tennessee-Martin": "UT Martin",
    "Illinois-Chicago": "UIC",
    "Wisconsin-Milwaukee": "Milwaukee",
    "Wisconsin-Green Bay": "Green Bay",
    "Missouri-Kansas City": "UMKC",
    "Loyola (IL)": "Loyola Chicago",
    "Loyola (MD)": "Loyola Maryland",
    "Mount St. Mary's": "Mount St. Mary's",
}


def normalize_team_name(name: str) -> str:
    """Strip footnote markers and apply canonical name fixes."""
    if not isinstance(name, str):
        return str(name)
    # Replace non-breaking spaces with regular spaces
    name = name.replace("\xa0", " ")
    # Strip trailing postseason labels SR appends (e.g. " NCAA", " NIT", " CBI")
    name = re.sub(r"\s+(NCAA|NIT|CBI|CIT|NEC|NAIA|NJCAA)$", "", name).strip()
    # Remove trailing footnote characters like *, †, ‡, ^
    name = re.sub(r"[\*†‡\^]+$", "", name).strip()
    return NAME_FIXES.get(name, name)
