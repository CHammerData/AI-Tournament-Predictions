"""
YAML writer — merges raw scraped JSON with draft prospect data and writes
final per-year YAML files to data/processed/{mens,womens}/{year}.yaml.

This is the final processing step before feature engineering. It:
  1. Loads raw JSON from data/raw/{mens,womens}/{year}.json
  2. Loads the draft prospect lookup from data/raw/draft_lookup.json
  3. Merges prospect counts into each team record
  4. Validates required fields, logs warnings for missing data
  5. Writes clean YAML files conforming to the schema in README.md

Usage:
    python processing/yaml_writer.py --gender mens
    python processing/yaml_writer.py --gender womens
    python processing/yaml_writer.py --gender both   # default
    python processing/yaml_writer.py --year 2023     # single year only
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DRAFT_LOOKUP_PATH = RAW_DIR / "draft_lookup.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("yaml_writer")

# Fields that must be present (non-None) for a team record to be valid
REQUIRED_FIELDS = ["seed", "region", "conference", "record"]

# Fields we warn about but don't discard records for
OPTIONAL_WARN_FIELDS = [
    "advanced.srs",
    "advanced.offensive_rating",
    "advanced.defensive_rating",
    "tournament_result.round_reached",
]


# ---------------------------------------------------------------------------
# Prospect lookup helpers
# ---------------------------------------------------------------------------

def load_draft_lookup() -> dict:
    """Load the NBA/WNBA draft prospect lookup dict."""
    if not DRAFT_LOOKUP_PATH.exists():
        logger.warning(
            "Draft lookup not found at %s — run nba_wnba_scraper.py first. "
            "Prospect counts will be None.",
            DRAFT_LOOKUP_PATH,
        )
        return {}
    with DRAFT_LOOKUP_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def get_prospect_count(
    draft_lookup: dict,
    league: str,
    team: str,
    year: int,
) -> Optional[int]:
    """Return the prospect count for (league, team, year), or None."""
    league_data = draft_lookup.get(league, {})
    team_data = league_data.get(team, {})
    count = team_data.get(str(year)) or team_data.get(year)
    return int(count) if count is not None else None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _get_nested(d: dict, dotted_key: str):
    """Retrieve a nested value using dot notation, e.g. 'advanced.srs'."""
    keys = dotted_key.split(".")
    val = d
    for k in keys:
        if not isinstance(val, dict):
            return None
        val = val.get(k)
    return val


def validate_record(team: str, record: dict, year: int) -> bool:
    """
    Check that required fields are present. Returns True if valid.
    Logs warnings for optional missing fields regardless.
    """
    valid = True
    for field in REQUIRED_FIELDS:
        val = _get_nested(record, field)
        if val is None:
            logger.warning("Year %d | %s | MISSING required field: %s", year, team, field)
            valid = False

    for field in OPTIONAL_WARN_FIELDS:
        val = _get_nested(record, field)
        if val is None:
            logger.debug("Year %d | %s | missing optional field: %s", year, team, field)

    return valid


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def enrich_and_clean(
    team_data: dict,
    draft_lookup: dict,
    gender: str,
) -> dict:
    """
    Merge draft prospect counts into a team record and clean up types.
    gender: "mens" or "womens"
    """
    team = team_data.get("team", "")
    year = team_data.get("year", 0)

    # Inject prospect count
    if gender == "mens":
        count = get_prospect_count(draft_lookup, "nba", team, year)
        team_data["nba_prospects"] = count
        # Remove womens key if present
        team_data.pop("wnba_prospects", None)
    else:
        count = get_prospect_count(draft_lookup, "wnba", team, year)
        team_data["wnba_prospects"] = count
        team_data.pop("nba_prospects", None)

    # Ensure numeric types are correct (may have been stringified in JSON)
    _coerce_types(team_data)

    return team_data


def _coerce_types(record: dict) -> None:
    """In-place type coercion for numeric fields."""
    int_fields = [("seed",), ("tournament_result", "wins"), ("tournament_result", "losses")]
    float_fields = [
        ("strength_of_schedule", "rating"),
        ("advanced", "srs"),
        ("advanced", "offensive_rating"),
        ("advanced", "defensive_rating"),
        ("advanced", "pace"),
        ("advanced", "efg_pct"),
        ("advanced", "tov_pct"),
        ("advanced", "orb_pct"),
        ("advanced", "ft_rate"),
    ]
    bool_fields = [
        ("conf_champion",),
        ("conf_tourn_champion",),
        ("tournament_result", "upset_caused"),
        ("tournament_result", "upset_suffered"),
    ]

    def set_nested(d, keys, val):
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val

    def get_nested(d, keys):
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return d

    for keys in int_fields:
        val = get_nested(record, keys)
        if val is not None:
            try:
                set_nested(record, keys, int(float(str(val))))
            except (ValueError, TypeError):
                pass

    for keys in float_fields:
        val = get_nested(record, keys)
        if val is not None:
            try:
                set_nested(record, keys, round(float(str(val)), 3))
            except (ValueError, TypeError):
                pass

    for keys in bool_fields:
        val = get_nested(record, keys)
        if val is not None:
            if isinstance(val, str):
                set_nested(record, keys, val.lower() in ("true", "1", "yes"))
            else:
                set_nested(record, keys, bool(val))


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------

def write_year_yaml(year: int, teams: list[dict], gender: str) -> Path:
    """Write a list of team dicts to data/processed/{gender}/{year}.yaml."""
    out_dir = PROCESSED_DIR / gender
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}.yaml"

    # Sort teams by seed for readability
    teams_sorted = sorted(teams, key=lambda t: (t.get("region", ""), t.get("seed", 99)))

    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            {"year": year, "tournament": gender, "teams": teams_sorted},
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    logger.info("Wrote %d teams -> %s", len(teams_sorted), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def process_gender(gender: str, year_filter: Optional[int] = None) -> None:
    """Process all raw JSON files for a given gender into YAML."""
    raw_dir = RAW_DIR / gender
    if not raw_dir.exists():
        logger.error("Raw directory not found: %s — run the scraper first.", raw_dir)
        return

    draft_lookup = load_draft_lookup()

    # Collect year files
    year_files = sorted(raw_dir.glob("[0-9][0-9][0-9][0-9].json"))
    if not year_files:
        logger.error("No year JSON files found in %s", raw_dir)
        return

    for json_path in year_files:
        year = int(json_path.stem)
        if year_filter is not None and year != year_filter:
            continue

        logger.info("Processing %s/%d", gender, year)
        with json_path.open(encoding="utf-8") as f:
            raw_data: dict = json.load(f)

        valid_teams = []
        skipped = 0

        for team_name, team_record in raw_data.items():
            enriched = enrich_and_clean(team_record, draft_lookup, gender)
            if validate_record(team_name, enriched, year):
                valid_teams.append(enriched)
            else:
                skipped += 1
                logger.debug("Skipping %s %d — failed validation", team_name, year)

        logger.info(
            "%s/%d: %d valid teams, %d skipped",
            gender, year, len(valid_teams), skipped
        )

        if valid_teams:
            write_year_yaml(year, valid_teams, gender)
        else:
            logger.warning("%s/%d: no valid teams — YAML not written.", gender, year)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Write processed YAML from raw scraped JSON")
    parser.add_argument(
        "--gender", choices=["mens", "womens", "both"], default="both",
        help="Which gender's data to process"
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Process only this specific year (default: all available years)"
    )
    args = parser.parse_args()

    genders = ["mens", "womens"] if args.gender == "both" else [args.gender]
    for gender in genders:
        process_gender(gender, year_filter=args.year)

    logger.info("YAML writing complete.")


if __name__ == "__main__":
    main()
