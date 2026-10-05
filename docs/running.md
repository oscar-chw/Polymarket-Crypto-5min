# Running the tests and the walk-forward (PowerShell)

The README's [Quick start](../README.md#quick-start) gives the same steps for bash. Raw and generated data are written
under the git-ignored `data/` folder.

```powershell
git clone https://github.com/oscar-chw/Polymarket-Crypto-5min.git
cd Polymarket-Crypto-5min
uv sync --python 3.12 --extra dev --frozen
uv run pytest -q

# Bounded public-data download for an API-shape check.
uv run python scripts/download_history.py --max-pages 2

# Full public-data acquisition and chronological base walk-forward.
uv run python scripts/run_btc_5m_full_history_walk_forward.py `
  --initial-capital 2000 `
  --stake 10 `
  --train-markets 400 `
  --test-markets 100 `
  --price-source both

# Exit-aware run reusing the full-history inputs.
uv run python scripts/exit_aware_walk_forward.py `
  --initial-capital 2000 `
  --stake 10 `
  --train-markets 400 `
  --test-markets 100 `
  --out-dir data/processed/exit_aware_walk_forward
```
