---
name: AI Tournament Predictions project overview
description: High-level goals, architecture, and build order for the NCAA bracket prediction project
type: project
---

Building a March Madness bracket prediction system (Men's + Women's) in Python.

Three main components:
1. **Scrapers** — historical data 2014–2024 from sports-reference.com/cbb, NCAA API, NBA/WNBA draft pages. Output: raw cache + processed YAML per team per year.
2. **Models** — separate Men's and Women's models (Logistic Regression + XGBoost ensemble, Platt-calibrated). Feature matrix uses per-matchup differentials. Leave-one-year-out CV for backtesting.
3. **Bracket generator** — Monte Carlo simulation (N=10k) → optimal bracket per ESPN scoring rules.

**Why:** User has not watched any college basketball and wants data/AI to make all picks.

**YAML schema** is defined in README.md — captures seed, conf champion flags, SOS, advanced metrics (SRS, ORtg, DRtg, Pace, eFG%), NBA/WNBA prospect count, coach experience, tournament result.

**How to apply:** When implementing scrapers, follow the field list from the README YAML schema exactly. Default to Python. Rate-limit scraping (3s sleep between requests on sports-reference). Cache raw data before processing so we don't re-scrape.
