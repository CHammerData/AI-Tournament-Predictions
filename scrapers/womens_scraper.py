"""
Women's NCAA tournament data scraper.

This script is a lightweight wrapper around the generic NCAAScraper that
is configured for the women's tournament.

Usage:
    python scrapers/womens_scraper.py --years 2014-2025
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
logger = logging.getLogger("womens_scraper")

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "womens"


def parse_years(years_arg: str) -> list[int]:
    if "-" in years_arg and "," not in years_arg:
        start, end = years_arg.split("-")
        return list(range(int(start), int(end) + 1))
    elif "," in years_arg:
        return [int(y.strip()) for y in years_arg.split(",")]
    else:
        return [int(years_arg.strip())]


def main():
    parser = argparse.ArgumentParser(description="Scrape women's NCAA tournament data")
    parser.add_argument("--years", default="2014-2025")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    years = parse_years(args.years)
    logger.info("Women's — scraping years: %s", years)

    scraper = NCAAScraper("womens")
    all_data = {}
    for year in years:
        year_data = scraper.scrape_year(year)
        scraper.save_raw(year, year_data)
        all_data[year] = year_data

    combined_path = OUTPUT_DIR / "all_years.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, default=str)
    logger.info("Women's all years saved -> %s", combined_path)


if __name__ == "__main__":
    main()