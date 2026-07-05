"""Project configuration defaults.

The defaults intentionally use only public market-data endpoints. Trading/order
execution is not implemented in this repository.
"""

from __future__ import annotations

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
DATA_API_BASE_URL = "https://data-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
BINANCE_BASE_URL = "https://api.binance.com"

# Current Polymarket crypto taker fee formula is:
#   fee_per_share = fee_rate * price * (1 - price)
# Makers are not charged by that formula, but this repo assumes taker execution
# for conservative research/backtesting.
DEFAULT_CRYPTO_TAKER_FEE_RATE = 0.07

DEFAULT_HTTP_TIMEOUT = 20
DEFAULT_MAX_RETRIES = 4
