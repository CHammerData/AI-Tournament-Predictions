# Implementation Plan

## Phase 1 — Data Collection (Scrapers)

### 1a. Men's Scraper (`scrapers/mens_scraper.py`)
- [ ] Scrape sports-reference.com/cbb for per-team season stats (2014–2024)
  - Advanced table: SRS, SOS, ORtg, DRtg, Pace, eFG%, TOV%, ORB%, FT rate
  - Summary table: overall record, conference record, seed
- [ ] Scrape NCAA tournament bracket results per year (round-by-round outcomes)
- [ ] Tag conference champions (regular season + conf tournament)
- [ ] Compute momentum (last 10 regular season games)
- [ ] Write raw cache to `data/raw/mens/`

### 1b. Women's Scraper (`scrapers/womens_scraper.py`)
- [ ] Mirror men's scraper logic against sports-reference.com/cbb/womens
- [ ] Note: Women's field expanded to 68 teams in 2022 — handle field-size changes
- [ ] Write raw cache to `data/raw/womens/`

### 1c. NBA/WNBA Prospect Scraper (`scrapers/nba_wnba_scraper.py`)
- [ ] Scrape NBA draft classes 2014–2027 (to cover prospects from 2024 rosters)
- [ ] Scrape WNBA draft classes 2014–2027
- [ ] For each team+year, count players drafted within 3 years of that season
- [ ] Output lookup dict: `{team: {year: prospect_count}}`

### 1d. YAML Writer (`processing/yaml_writer.py`)
- [ ] Merge scraped data into YAML schema (see README)
- [ ] One YAML file per tournament year per gender: `data/processed/mens/2023.yaml`
- [ ] Validate required fields; log missing data warnings

---

## Phase 2 — Feature Engineering & Trend Analysis

### 2a. Feature Engineering (`processing/feature_engineering.py`)
- [ ] Load all YAML files into a pandas DataFrame
- [ ] Create matchup rows: for each historical game, compute Team A - Team B feature differentials
  - seed_diff, srs_diff, sos_diff, ortg_diff, drtg_diff, prospect_diff, etc.
  - Binary flags: conf_champion, momentum_positive
- [ ] Label each row with outcome (1 = team A won, 0 = team A lost)
- [ ] Export `data/processed/mens_matchups.csv` and `data/processed/womens_matchups.csv`

### 2b. Trend Analysis (`processing/trend_analysis.py`)
- [ ] Historical win rates by seed pairing (all 1v16 through 8v9 etc.)
- [ ] Upset rate by round
- [ ] Conference win rates by round
- [ ] Correlation matrix: features vs. win outcome
- [ ] Save summary to `data/trends/mens_trends.yaml` and `womens_trends.yaml`
- [ ] Generate plots saved to `data/trends/plots/`

---

## Phase 3 — Model Training

### 3a. Base Model (`models/train.py`)
- [ ] Load matchup CSVs
- [ ] Train/test split: leave-one-year-out cross-validation
- [ ] Men's model pipeline:
  - Logistic Regression (interpretable baseline)
  - XGBoost classifier
  - Ensemble via soft voting / stacking
  - Platt scaling calibration
- [ ] Women's model: same pipeline, fit separately
- [ ] Save models to `models/saved/mens_model.pkl` and `womens_model.pkl`
- [ ] Log feature importances

### 3b. Evaluation (`models/evaluate.py`)
- [ ] Brier score, log-loss, accuracy per round
- [ ] Compare vs. seed-only baseline
- [ ] Upset detection precision/recall
- [ ] Print backtest bracket accuracy (% of Final Four correct, champion correct)

---

## Phase 4 — Bracket Generation

### 4a. Simulation (`bracket/simulate.py`)
- [ ] Load current-year team YAML (scraped fresh for 2025)
- [ ] Build bracket structure (64/68 team bracket object)
- [ ] For each round, compute win probabilities for all live matchups
- [ ] Monte Carlo simulate N=10,000 full brackets
- [ ] Aggregate: for each team, P(reaching each round)
- [ ] Select "optimal" bracket maximizing expected ESPN scoring

### 4b. Output (`bracket/output_bracket.py`)
- [ ] Print bracket to console (ASCII art or tabular)
- [ ] Save JSON: `bracket/brackets/mens_2025.json`
- [ ] Save CSV: `bracket/brackets/mens_2025.csv`

---

## Data Source Notes

| Source | Rate Limits / Notes |
|---|---|
| sports-reference.com | ~20 req/min polite limit; use `time.sleep(3)` between requests |
| NCAA.com | Public JSON API endpoints available for bracket data |
| ESPN (unofficial) | Undocumented API — use with care, may break |
| NBA.com | Requires headers to avoid 403; use known working headers |

## Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Website scraping breaks | Cache raw HTML/JSON; scrape once, process many times |
| Small sample size (10 years × ~68 teams) | Use feature differentials to 2x data; strong regularization |
| Women's data availability | Her Hoop Stats as backup; manual data gap filling |
| Overfitting to upsets | Calibrate probabilities; don't optimize for upsets specifically |
