"""
NBA and WNBA draft prospect scraper.

Scrapes Wikipedia's NBA/WNBA draft class pages to build a lookup of how many
players each college sent to the draft within 3 years of a tournament season.

Why Wikipedia: sports-reference is on a separate domain (basketball-reference)
with aggressive Cloudflare protection. NBA.com API also blocks automated
requests at scale. Wikipedia is open, stable, and has complete draft data.

Output:
  data/raw/nba_draft/{year}.json     — college picks per draft year
  data/raw/wnba_draft/{year}.json    — college picks per draft year
  data/raw/draft_lookup.json         — aggregated prospect counts:
    {"nba": {team: {year: count}}, "wnba": {team: {year: count}}}

Usage:
    python scrapers/nba_wnba_scraper.py
    python scrapers/nba_wnba_scraper.py --min-year 2014 --max-year 2025
"""

import argparse
import io
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from base_scraper import normalize_team_name, PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("draft_scraper")

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
REQUEST_DELAY = 1.5  # seconds between Wikipedia requests
PROSPECT_WINDOW = 3  # years forward to credit a prospect to prior seasons

WIKI_HEADERS = {
    "User-Agent": (
        "TournamentPredictionBot/1.0 "
        "(educational project; contact via GitHub AI-Tournament-Predictions)"
    )
}

# College-year suffix patterns in Wikipedia: "Alabama (Fr.)", "Duke (So.)", etc.
CLASS_SUFFIX = re.compile(r"\s*\([A-Za-z]+\.?\)$")

# Strings that indicate a non-college entry (international clubs, G-League, etc.)
NON_COLLEGE_PATTERNS = re.compile(
    r"\(France\)|\(Spain\)|\(Australia\)|\(Turkey\)|\(Serbia\)|G League|Ignite|"
    r"Overtime Elite|G-League|\(Argentina\)|\(Lithuania\)|\(Latvia\)|"
    r"\(Germany\)|\(Greece\)|\(Canada\)|\(Brazil\)|\(Israel\)|"
    r"high school|prep school|undrafted|not applicable",
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Wikipedia scraper
# ---------------------------------------------------------------------------

def _wiki_url(draft_year: int, league: str) -> str:
    if league == "nba":
        return f"https://en.wikipedia.org/wiki/{draft_year}_NBA_draft"
    else:
        return f"https://en.wikipedia.org/wiki/{draft_year}_WNBA_draft"


def _cache_path(draft_year: int, league: str) -> Path:
    subdir = "nba_draft" if league == "nba" else "wnba_draft"
    return OUTPUT_DIR / subdir / f"{draft_year}.json"


def fetch_draft_picks(draft_year: int, league: str) -> list[dict]:
    """
    Scrape one season's draft picks from Wikipedia.

    Returns a list of dicts: [{player, college, draft_year}, ...]
    Only college players are included (international/G-League filtered out).
    Results are cached to disk.
    """
    cache = _cache_path(draft_year, league)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        logger.debug("Cache hit: %s draft %d", league.upper(), draft_year)
        with cache.open(encoding="utf-8") as f:
            return json.load(f)

    url = _wiki_url(draft_year, league)
    logger.info("Fetching %s draft %d from Wikipedia", league.upper(), draft_year)

    try:
        r = requests.get(url, headers=WIKI_HEADERS, timeout=20)
        r.raise_for_status()
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning("%s draft %d page not found on Wikipedia", league.upper(), draft_year)
        else:
            logger.error("Wikipedia fetch failed (%s %d): %s", league.upper(), draft_year, e)
        cache.write_text("[]", encoding="utf-8")
        return []
    except Exception as e:
        logger.error("Wikipedia fetch failed (%s %d): %s", league.upper(), draft_year, e)
        cache.write_text("[]", encoding="utf-8")
        return []

    time.sleep(REQUEST_DELAY)

    picks = _parse_wiki_draft(r.text, draft_year, league)
    cache.write_text(json.dumps(picks, indent=2), encoding="utf-8")
    logger.info("  → %d college picks for %s %d", len(picks), league.upper(), draft_year)
    return picks


def _parse_wiki_draft(html: str, draft_year: int, league: str) -> list[dict]:
    """
    Parse the Wikipedia draft page HTML to extract college players.

    Wikipedia draft tables have columns like:
      Rnd. | Pick | Player | Pos. | Nationality | Team | School / club team

    The school column contains entries like:
      "Alabama (Fr.)"    → college: Alabama
      "Duke (So.)"       → college: Duke
      "Metropolitans 92 (France)"  → international, skip
    """
    try:
        dfs = pd.read_html(io.StringIO(html))
    except Exception as e:
        logger.error("Could not parse Wikipedia tables: %s", e)
        return []

    picks = []

    for df in dfs:
        # Find the table that looks like a draft table: has Player + school column
        cols_lower = [str(c).lower() for c in df.columns]
        has_player = any("player" in c for c in cols_lower)
        has_school = any("school" in c or "club" in c for c in cols_lower)

        if not (has_player and has_school):
            continue

        # Identify the school column
        school_col = next(
            (c for c in df.columns if "school" in str(c).lower() or "club" in str(c).lower()),
            None
        )
        player_col = next(
            (c for c in df.columns if "player" in str(c).lower()),
            None
        )

        if school_col is None or player_col is None:
            continue

        for _, row in df.iterrows():
            school_raw = str(row[school_col]) if pd.notna(row[school_col]) else ""
            player_raw = str(row[player_col]) if pd.notna(row[player_col]) else ""

            if not school_raw or school_raw in ("nan", "—", ""):
                continue

            # Skip non-college entries
            if NON_COLLEGE_PATTERNS.search(school_raw):
                continue

            # Strip class year suffix: "Alabama (Fr.)" → "Alabama"
            college = CLASS_SUFFIX.sub("", school_raw).strip()

            # Strip Wikipedia footnote markers (~, +, #, *)
            player = re.sub(r"[~\+\#\*]+$", "", player_raw).strip()
            college = re.sub(r"[~\+\#\*]+$", "", college).strip()

            if not college or not player:
                continue

            picks.append({
                "player": player,
                "college": normalize_team_name(college),
                "draft_year": draft_year,
            })

    return picks


# ---------------------------------------------------------------------------
# Build prospect lookup
# ---------------------------------------------------------------------------

def build_prospect_lookup(tourney_years: list[int], league: str) -> dict:
    """
    Build {college_team: {tourney_year: prospect_count}}.

    Credits players back to prior tournament seasons within PROSPECT_WINDOW:
      1-and-done: draft_year D → tourney_year D only
      Senior:     draft_year D → tourney years D-3..D
    """
    min_draft = min(tourney_years)
    max_draft = max(tourney_years) + PROSPECT_WINDOW
    draft_years = list(range(min_draft, max_draft + 1))

    all_picks: list[dict] = []
    for dy in draft_years:
        picks = fetch_draft_picks(dy, league)
        all_picks.extend(picks)

    if not all_picks:
        logger.warning("No draft picks found for %s", league.upper())
        return {}

    lookup: dict = defaultdict(lambda: defaultdict(int))

    for pick in all_picks:
        college = pick.get("college")
        draft_year = pick.get("draft_year")
        if not college or not draft_year:
            continue

        for offset in range(0, PROSPECT_WINDOW + 1):
            tourney_year = draft_year - offset
            if tourney_year in tourney_years:
                lookup[college][tourney_year] += 1

    return {team: dict(years) for team, years in lookup.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape NBA/WNBA draft prospect data from Wikipedia")
    parser.add_argument("--min-year", type=int, default=2014)
    parser.add_argument("--max-year", type=int, default=2025)
    args = parser.parse_args()

    tourney_years = list(range(args.min_year, args.max_year + 1))
    logger.info("Building prospect lookups for tourney years %s", tourney_years)

    nba_lookup = build_prospect_lookup(tourney_years, league="nba")
    wnba_lookup = build_prospect_lookup(tourney_years, league="wnba")

    output = {"nba": nba_lookup, "wnba": wnba_lookup}
    out_path = OUTPUT_DIR / "draft_lookup.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(
        "Draft lookup saved -> %s  (NBA colleges: %d, WNBA colleges: %d)",
        out_path, len(nba_lookup), len(wnba_lookup)
    )

    # Print a sample to verify
    sample_teams = ["Duke", "UConn", "Kentucky", "Kansas"]
    for team in sample_teams:
        nba = nba_lookup.get(team, {})
        wnba = wnba_lookup.get(team, {})
        if nba or wnba:
            logger.info("  %s → NBA: %s | WNBA: %s", team, dict(nba), dict(wnba))


if __name__ == "__main__":
    main()