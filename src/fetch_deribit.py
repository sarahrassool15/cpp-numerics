# fetch_deribit.py
# Takes a snapshot of every Bitcoin option on Deribit (the biggest crypto options exchange)
# from its free public API, and keeps one expiry to study.
#
# Deribit quirks I need to handle:
#   - option prices are quoted in BTC, not dollars, so dollar price = price in BTC * BTC price
#   - each option comes with "underlying_price", the forward price for its expiry,
#     which is the right price to plug into the option formula
#   - instrument names look like BTC-27MAR26-100000-C  (expiry, strike, Call or Put)

from datetime import datetime, timezone

import pandas as pd
import requests

URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
resp = requests.get(URL, params={"currency": "BTC", "kind": "option"}, timeout=20)
resp.raise_for_status()
rows = resp.json()["result"]

now = datetime.now(timezone.utc)
records = []
for r in rows:
    _, expiry, strike, kind = r["instrument_name"].split("-")
    if r.get("mark_price") is None or r.get("underlying_price") is None:
        continue
    # Deribit options expire at 08:00 UTC on the expiry date
    exp = datetime.strptime(expiry, "%d%b%y").replace(hour=8, tzinfo=timezone.utc)
    records.append({
        "instrument": r["instrument_name"],
        "expiry": exp,
        "T": (exp - now).total_seconds() / (365.25 * 24 * 3600),   # years to expiry
        "strike": float(strike),
        "type": kind,                                   # "C" or "P"
        "forward": r["underlying_price"],
        "price_usd": r["mark_price"] * r["underlying_price"],
        "deribit_iv": r.get("mark_iv"),                 # Deribit's own implied vol, in %
        "open_interest": r.get("open_interest", 0),
    })

df = pd.DataFrame(records)
df["snapshot_utc"] = now
df.to_csv("data/deribit_all.csv", index=False)

# choose ONE expiry: the one between 3 and 8 weeks away with the most strikes
# NOTE: very short expiries are dominated by tiny time values, very long ones are thinly traded
candidates = df[(df["T"] > 21 / 365) & (df["T"] < 56 / 365)]
best = candidates.groupby("expiry")["strike"].nunique().idxmax()
one = df[df["expiry"] == best].copy()

# keep out-of-the-money options only: puts below the forward, calls above.
# These are the ones that actually trade, and their prices are almost all "time value",
# which is what carries the information about volatility
F = one["forward"].median()
otm = one[((one["type"] == "P") & (one["strike"] < F)) | ((one["type"] == "C") & (one["strike"] >= F))].copy()
otm = otm[otm["price_usd"] > 0].sort_values("strike")
otm.to_csv("data/deribit_expiry.csv", index=False)

print(f"snapshot taken {now:%Y-%m-%d %H:%M} UTC, {len(df):,} BTC options in total")
print(f"chosen expiry {best:%d %b %Y}, {otm['T'].iloc[0] * 365:.0f} days away, forward price ${F:,.0f}")
print(f"{len(otm)} out-of-the-money options, strikes ${otm['strike'].min():,.0f} to ${otm['strike'].max():,.0f}")
