"""
Men's NCAA tournament data scraper.

This script is a lightweight wrapper around the generic NCAAScraper that
is configured for the men's tournament.

Usage:
    python scrapers/mens_scraper.py --years 2014-2024
    python scrapers/mens_scraper.py --years 2023      # single year
    python scrapers/mens_scraper.py --years 2022,2023 # specific years
"""

import argparse
import json
import logging

from ncaa_scraper import NCAAScraper
from base_scraper import PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("mens_scraper")

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "mens"


def parse_years(years_arg: str) -> list[int]:
    if "-" in years_arg and "," not in years_arg:
        start, end = years_arg.split("-")
        return list(range(int(start), int(end) + 1))
    elif "," in years_arg:
        return [int(y.strip()) for y in years_arg.split(",")]
    else:
        return [int(years_arg.strip())]


def main():
    parser = argparse.ArgumentParser(description="Scrape men's NCAA tournament data")
    parser.add_argument(
        "--years", default="2014-2024",
        help="Year range (2014-2024), comma list (2022,2023), or single year",
    )
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    years = parse_years(args.years)
    logger.info("Scraping years: %s", years)

    scraper = NCAAScraper("mens")
    all_data = {}
    for year in years:
        year_data = scraper.scrape_year(year)
        scraper.save_raw(year, year_data)
        all_data[year] = year_data

    combined_path = OUTPUT_DIR / "all_years.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, default=str)
    logger.info("All years saved -> %s", combined_path)


if __name__ == "__main__":
    main()
