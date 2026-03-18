#!/usr/bin/env python3
"""
2026 Mock Draft Scraper

Scrapes current NBA and WNBA mock drafts from Tankathon to build prospect
counts for teams in the 2026 tournament. Used as a proxy for the real draft
since it hasn't happened yet.

Source: Tankathon.com (same parsing logic for NBA and WNBA)
  NBA:  https://www.tankathon.com/mock_draft
  WNBA: https://www.tankathon.com/wnba/mock_draft

Adds year "2026" entries to data/raw/draft_lookup.json without overwriting
existing historical data.

Usage:
    python scrapers/mock_draft_scraper.py
    python scrapers/mock_draft_scraper.py --nba-only
    python scrapers/mock_draft_scraper.py --wnba-only
    python scrapers/mock_draft_scraper.py --dry-run   # print picks, don't save
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from base_scraper import normalize_team_name
except ImportError:
    def normalize_team_name(name: str) -> str:
        return name.strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
)
logger = logging.getLogger("mock_draft_scraper")

PROJECT_ROOT = Path(__file__).parent.parent
DRAFT_LOOKUP = PROJECT_ROOT / "data" / "raw" / "draft_lookup.json"
DRAFT_YEAR   = 2026

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}

TANKATHON_URLS = {
    "nba":  "https://www.tankathon.com/mock_draft",
    "wnba": "https://www.tankathon.com/wnba/mock_draft",
}

# Row format: "{pick} {player} {pos} | {college} {height} ..."
# The pipe character reliably separates pos from college on Tankathon.
ROW_RE = re.compile(r"^(\d+)\s+(.+?)\s+[\w/]+\s*\|\s*(.+?)\s+\d")

# ── Scraper ────────────────────────────────────────────────────────────────────

def fetch_mock_draft(league: str) -> list[dict]:
    """
    Scrape Tankathon mock draft for the given league ('nba' or 'wnba').
    Returns list of {pick, player, college, draft_year} dicts.
    """
    url = TANKATHON_URLS[league]
    logger.info("Fetching %s mock draft from %s", league.upper(), url)

    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        logger.error("Failed to fetch %s mock draft: %s", league.upper(), e)
        return []

    time.sleep(1.5)

    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.find_all("div", class_="mock-row")
    if not rows:
        logger.warning("No mock-row divs found on %s — page layout may have changed", url)
        return []

    picks = []
    for row in rows:
        text = row.get_text(" ", strip=True)
        m = ROW_RE.match(text)
        if not m:
            continue

        pick_num = int(m.group(1))
        player   = m.group(2).strip()
        college  = m.group(3).strip()

        # Skip non-college entries (international players show club names)
        if any(skip in college for skip in ("Valencia", "Madrid", "Barcelona",
                                             "Overtime", "G League", "Ignite")):
            logger.debug("  Skipping non-college pick %d: %s (%s)", pick_num, player, college)
            continue

        picks.append({
            "pick":       pick_num,
            "player":     player,
            "college":    normalize_team_name(college),
            "draft_year": DRAFT_YEAR,
        })

    logger.info("  Parsed %d %s mock draft picks", len(picks), league.upper())
    return picks

# ── Prospect count aggregation ─────────────────────────────────────────────────

def picks_to_college_counts(picks: list[dict]) -> dict[str, int]:
    """Count projected draft picks per college team."""
    counts: dict[str, int] = {}
    for pick in picks:
        college = pick.get("college", "")
        if college:
            counts[college] = counts.get(college, 0) + 1
    return counts

# ── Update draft_lookup.json ───────────────────────────────────────────────────

def update_draft_lookup(
    nba_picks: list[dict],
    wnba_picks: list[dict],
    dry_run: bool = False,
) -> None:
    """
    Add year=2026 entries to draft_lookup.json.
    Existing historical data is preserved; only 2026 keys are added/updated.
    """
    if DRAFT_LOOKUP.exists():
        with DRAFT_LOOKUP.open(encoding="utf-8") as f:
            lookup = json.load(f)
    else:
        lookup = {"nba": {}, "wnba": {}}

    year_key = str(DRAFT_YEAR)
    nba_counts  = picks_to_college_counts(nba_picks)
    wnba_counts = picks_to_college_counts(wnba_picks)

    for college, count in nba_counts.items():
        lookup["nba"].setdefault(college, {})[year_key] = count

    for college, count in wnba_counts.items():
        lookup["wnba"].setdefault(college, {})[year_key] = count

    if dry_run:
        logger.info("[DRY RUN] Would write %d NBA + %d WNBA college entries for %d",
                    len(nba_counts), len(wnba_counts), DRAFT_YEAR)
        return

    with DRAFT_LOOKUP.open("w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2)

    logger.info("draft_lookup.json updated: %d NBA + %d WNBA colleges added for %d",
                len(nba_counts), len(wnba_counts), DRAFT_YEAR)

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape 2026 NBA/WNBA mock drafts from Tankathon")
    parser.add_argument("--nba-only",  action="store_true")
    parser.add_argument("--wnba-only", action="store_true")
    parser.add_argument("--dry-run",   action="store_true", help="Print results without saving")
    args = parser.parse_args()

    nba_picks  = []
    wnba_picks = []

    if not args.wnba_only:
        nba_picks = fetch_mock_draft("nba")
        nba_counts = picks_to_college_counts(nba_picks)
        print(f"\nNBA 2026 mock draft — {len(nba_picks)} picks across {len(nba_counts)} colleges")
        print(f"{'Pick':>4}  {'Player':<25}  College")
        print("-" * 55)
        for p in nba_picks:
            print(f"  {p['pick']:>2}.  {p['player']:<25}  {p['college']}")
        print(f"\nProspects per college (top 10):")
        for college, count in sorted(nba_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {college:<30} {count}")

    if not args.nba_only:
        wnba_picks = fetch_wnba_mock_draft()
        wnba_counts = picks_to_college_counts(wnba_picks)
        print(f"\nWNBA 2026 mock draft — {len(wnba_picks)} picks across {len(wnba_counts)} colleges")
        print(f"{'Pick':>4}  {'Player':<25}  College")
        print("-" * 55)
        for p in wnba_picks:
            print(f"  {p['pick']:>2}.  {p['player']:<25}  {p['college']}")
        print(f"\nProspects per college (top 10):")
        for college, count in sorted(wnba_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {college:<30} {count}")

    if nba_picks or wnba_picks:
        update_draft_lookup(nba_picks, wnba_picks, dry_run=args.dry_run)
    else:
        logger.warning("No picks scraped — draft_lookup.json not updated")


def fetch_wnba_mock_draft() -> list[dict]:
    """Wrapper that calls fetch_mock_draft for WNBA."""
    return fetch_mock_draft("wnba")


if __name__ == "__main__":
    main()