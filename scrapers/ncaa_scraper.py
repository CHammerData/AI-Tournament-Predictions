"""
Gender-neutral NCAA tournament data scraper.

Pulls per-team season stats, advanced metrics, tournament results, and
conference champion flags from sports-reference.com/cbb. This script is
designed to be generic for both men's and women's tournaments.

Usage (from other scripts):
    from ncaa_scraper import NCAAScraper
    scraper = NCAAScraper("mens")
    scraper.scrape_year(2023)
"""

import json
import logging
import re
from typing import Optional

import pandas as pd

from base_scraper import (
    fetch,
    get_soup,
    normalize_team_name,
    parse_sr_table,
    PROJECT_ROOT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ncaa_scraper")


class NCAAScraper:
    """
    A generic scraper for NCAA tournament data for a given gender.
    """

    def __init__(self, gender: str):
        if gender not in ["mens", "womens"]:
            raise ValueError("Gender must be 'mens' or 'womens'")
        self.gender = gender
        self.base = "https://www.sports-reference.com"
        self.adv_stats_url = self.base + f"/cbb/seasons/{self._gender_path()}/{{year}}-advanced-school-stats.html"
        self.standings_url = self.base + f"/cbb/seasons/{self._gender_path()}/{{year}}-standings.html"
        self.tournament_url = self.base + f"/cbb/postseason/{self._gender_path()}/{{year}}-ncaa.html"
        self.output_dir = PROJECT_ROOT / "data" / "raw" / self._gender_path()
        self.round_map = {
            0: "First Round",
            1: "Second Round",
            2: "Sweet Sixteen",
            3: "Elite Eight",
            4: "Final Four",
            5: "Championship Game",
            6: "Champion",
        }

    def _gender_path(self) -> str:
        return "men" if self.gender == "mens" else "women"

    def scrape_standings(self, year: int) -> dict:
        url = self.standings_url.format(year=year)
        try:
            html = fetch(url, cache_subdir=self.gender)
        except Exception as e:
            logger.warning("Could not fetch standings for %d: %s", year, e)
            return {}

        soup = get_soup(html)
        result: dict = {}

        for table in soup.find_all("table", id=re.compile(r"^standings_")):
            for row in table.select("tbody tr"):
                school_cell = row.find("td", {"data-stat": "school_name"})
                conf_cell = row.find("td", {"data-stat": "conf_abbr"})
                notes_cell = row.find("td", {"data-stat": "notes"})
                if school_cell is None:
                    continue
                raw_name = school_cell.get_text()
                school = normalize_team_name(re.sub(r"[\*\^]+", "", raw_name).strip())
                conf = conf_cell.get_text(strip=True) if conf_cell else ""
                notes = notes_cell.get_text(strip=True) if notes_cell else ""
                result[school] = {
                    "conf": conf,
                    "conf_champion": "Reg. Season Champion" in notes,
                    "conf_tourn_champion": "Conf. Tournament Champion" in notes,
                }

        logger.info("Year %d: conference data for %d schools from standings", year, len(result))
        return result

    def scrape_all_stats(self, year: int) -> Optional[pd.DataFrame]:
        url = self.adv_stats_url.format(year=year)
        html = fetch(url, cache_subdir=self.gender)
        df = parse_sr_table(html, "adv_school_stats")

        if df is None:
            logger.error("Could not parse adv_school_stats for year %d", year)
            return None

        df = self._normalise_columns(df)

        if "School" not in df.columns:
            logger.error("School column not found after normalisation for year %d. Cols: %s", year, list(df.columns))
            return None

        df["School"] = df["School"].apply(normalize_team_name)

        standings = self.scrape_standings(year)
        df["Conf"] = df["School"].map(lambda s: standings.get(s, {}).get("conf"))
        df["conf_champion"] = df["School"].map(
            lambda s: standings.get(s, {}).get("conf_champion", False)
        )
        df["conf_tourn_champion"] = df["School"].map(
            lambda s: standings.get(s, {}).get("conf_tourn_champion", False)
        )
        df["year"] = year

        logger.info("Year %d: %d teams scraped", year, len(df))
        return df

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename = {}
        for col in df.columns:
            c = col.lower()
            if re.search(r"_school$|^school$", c):
                rename[col] = "School"
            elif re.fullmatch(r"overall_w", c):
                rename[col] = "W"
            elif re.fullmatch(r"overall_l", c):
                rename[col] = "L"
            elif re.fullmatch(r"overall_srs", c):
                rename[col] = "SRS"
            elif re.fullmatch(r"overall_sos", c):
                rename[col] = "SOS"
            elif re.fullmatch(r"conf\._w", c):
                rename[col] = "W_conf"
            elif re.fullmatch(r"conf\._l", c):
                rename[col] = "L_conf"
            elif re.search(r"advanced_ortg|advanced_off_rtg", c):
                rename[col] = "ORtg"
            elif re.search(r"advanced_drtg|advanced_def_rtg", c):
                rename[col] = "DRtg"
            elif re.search(r"advanced_pace", c):
                rename[col] = "Pace"
            elif re.search(r"advanced_ftr", c):
                rename[col] = "FTr"
            elif re.search(r"advanced_3par", c):
                rename[col] = "3PAr"
            elif re.search(r"advanced_ts%", c):
                rename[col] = "TS_pct"
            elif re.search(r"advanced_efg%", c):
                rename[col] = "eFG_pct"
            elif re.search(r"advanced_tov%", c):
                rename[col] = "TOV_pct"
            elif re.search(r"advanced_orb%", c):
                rename[col] = "ORB_pct"
            elif re.search(r"advanced_ft/fga", c):
                rename[col] = "FT_per_FGA"

        return df.rename(columns=rename)

    def scrape_tournament_results(self, year: int) -> dict:
        url = self.tournament_url.format(year=year)
        try:
            html = fetch(url, cache_subdir=self.gender)
        except Exception as e:
            logger.error("Failed to fetch tournament page for %d: %s", year, e)
            return {}

        soup = get_soup(html)
        brackets_div = soup.find("div", id="brackets")
        if brackets_div is None:
            logger.warning("Year %d: no #brackets div found", year)
            return {}

        wins: dict[str, int] = {}
        seeds: dict[str, int] = {}
        region_map: dict[str, str] = {}
        upsets_caused: dict[str, bool] = {}
        upsets_suffered: dict[str, bool] = {}

        region_ids = ["east", "midwest", "south", "west"]

        for region_div in brackets_div.find_all("div", recursive=False):
            region_id = region_div.get("id", "")
            if region_id == "national":
                continue
            bracket = region_div.find("div", id="bracket")
            if not bracket:
                continue
            
            region_name = region_id.capitalize()
            self._parse_bracket(bracket, region_name, wins, seeds, region_map,
                           upsets_caused, upsets_suffered)

        national_div = brackets_div.find("div", id="national")
        if national_div:
            national_bracket = national_div.find("div", id="bracket")
            if national_bracket:
                self._parse_bracket(national_bracket, None, wins, seeds, region_map,
                               upsets_caused, upsets_suffered)

        results = {}
        for team, win_count in wins.items():
            seed = seeds.get(team, 0)
            results[team] = {
                "seed": seed,
                "region": region_map.get(team),
                "round_reached": self.round_map.get(win_count, f"Round {win_count}"),
                "wins": win_count,
                "losses": 0 if win_count == 6 else 1,
                "upset_caused": upsets_caused.get(team, False),
                "upset_suffered": upsets_suffered.get(team, False),
            }

        logger.info("Year %d: %d teams parsed from bracket", year, len(results))
        return results

    def _parse_bracket(
        self,
        bracket_div,
        region_name: Optional[str],
        wins: dict,
        seeds: dict,
        region_map: dict,
        upsets_caused: dict,
        upsets_suffered: dict,
    ) -> None:
        round_divs = bracket_div.find_all("div", class_="round", recursive=False)

        for round_div in round_divs:
            game_divs = round_div.find_all("div", recursive=False)

            for game_div in game_divs:
                team_divs = game_div.find_all("div", recursive=False)
                if len(team_divs) < 1:
                    continue

                teams_in_game = []
                for td in team_divs:
                    name, seed = self._extract_team(td)
                    if name:
                        won = "winner" in (td.get("class") or [])
                        teams_in_game.append((name, seed, won))

                for name, seed, _ in teams_in_game:
                    seeds[name] = seed
                    wins.setdefault(name, 0)
                    if region_name and name not in region_map:
                        region_map[name] = region_name

                if len(teams_in_game) == 2:
                    (a_name, a_seed, a_won), (b_name, b_seed, b_won) = teams_in_game
                    if a_won:
                        wins[a_name] = wins.get(a_name, 0) + 1
                        if a_seed > b_seed:
                            upsets_caused[a_name] = True
                        if b_seed < a_seed:
                            upsets_suffered.setdefault(b_name, False)
                    elif b_won:
                        wins[b_name] = wins.get(b_name, 0) + 1
                        if b_seed > a_seed:
                            upsets_caused[b_name] = True
                        if a_seed < b_seed:
                            upsets_suffered.setdefault(a_name, False)

    def _extract_team(self, div) -> tuple[str, int]:
        link = div.find("a", href=re.compile(r"/cbb/schools/"))
        if not link:
            return "", 0
        team = normalize_team_name(link.get_text(strip=True))
        span = div.find("span")
        seed = 0
        if span:
            try:
                seed = int(span.get_text(strip=True))
            except ValueError:
                seed = 0
        return team, seed

    def scrape_year(self, year: int) -> dict:
        logger.info("=== Scraping year %d for %s ===", year, self.gender)

        stats_df = self.scrape_all_stats(year)
        tournament_results = self.scrape_tournament_results(year)

        if stats_df is None:
            logger.error("Year %d: no stats, skipping.", year)
            return {}

        if not tournament_results:
            logger.warning("Year %d: no tournament results (cancelled or unavailable). Skipping.", year)
            return {}

        tournament_teams = set(tournament_results.keys())
        stats_df = stats_df[stats_df["School"].isin(tournament_teams)].copy()
        if stats_df.empty:
            logger.warning(
                "Year %d: 0 stats rows matched tournament teams. "
                "Tournament teams sample: %s. Stats schools sample: %s",
                year,
                list(tournament_teams)[:5],
                list(stats_df["School"].head() if not stats_df.empty else []),
            )

        def safe_float(val):
            try:
                return round(float(str(val)), 3)
            except (ValueError, TypeError):
                return None

        def safe_int(val):
            try:
                return int(float(str(val)))
            except (ValueError, TypeError):
                return None

        def safe_record(row, w_col, l_col):
            w = safe_int(row.get(w_col))
            l = safe_int(row.get(l_col))
            if w is not None and l is not None:
                return f"{w}-{l}"
            return None

        teams = {}
        for _, row in stats_df.iterrows():
            school = row["School"]
            tourn = tournament_results.get(school, {})

            teams[school] = {
                "year": year,
                "tournament": self.gender,
                "team": school,
                "seed": tourn.get("seed"),
                "region": tourn.get("region"),
                "conference": str(row.get("Conf", "") or "").strip() or None,
                "conf_champion": bool(row.get("conf_champion", False)),
                "conf_tourn_champion": bool(row.get("conf_tourn_champion", False)),
                "record": {
                    "overall": safe_record(row, "W", "L"),
                    "conference": safe_record(row, "W_conf", "L_conf"),
                },
                "strength_of_schedule": {
                    "rating": safe_float(row.get("SOS")),
                },
                "advanced": {
                    "srs": safe_float(row.get("SRS")),
                    "offensive_rating": safe_float(row.get("ORtg")),
                    "defensive_rating": safe_float(row.get("DRtg")),
                    "pace": safe_float(row.get("Pace")),
                    "efg_pct": safe_float(row.get("eFG_pct")),
                    "tov_pct": safe_float(row.get("TOV_pct")),
                    "orb_pct": safe_float(row.get("ORB_pct")),
                    "ft_rate": safe_float(row.get("FTr")),
                },
                "momentum": {
                    "last_10": None,
                },
                "nba_prospects": None,
                "coach": {
                    "name": None,
                    "years_experience": None,
                    "prior_tournament_appearances": None,
                },
                "tournament_result": {
                    "round_reached": tourn.get("round_reached"),
                    "wins": tourn.get("wins"),
                    "losses": tourn.get("losses"),
                    "upset_caused": tourn.get("upset_caused"),
                    "upset_suffered": tourn.get("upset_suffered"),
                },
            }

        logger.info("Year %d: %d tournament teams assembled.", year, len(teams))
        return teams

    def save_raw(self, year: int, data: dict) -> None:
        out_path = self.output_dir / f"{year}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Saved raw data -> %s", out_path)
