import time
import datetime as dt
import os
import math
from pathlib import Path
from zoneinfo import ZoneInfo
import traceback

import numpy as np
import pandas as pd
import yfinance as yf

import pyarrow as pa
import pyarrow.parquet as pq


# ==========================
# CONFIG
# ==========================
TICKER = "SPY"
VIX_TICKER = "^VIX"

REFRESH_SECONDS = 20

# Flow summary aggregation window (strike +/- around spot)
STRIKE_WINDOW_FLOW = 15

# Candidate contract selection window around spot
STRIKE_WINDOW_CANDIDATES = 25

# How many contracts to store per snapshot (calls & puts)
TOP_N_CALLS = 18
TOP_N_PUTS = 18

# Filters
MIN_MID_TO_COUNT = 0.01
MIN_VOLUME_TO_COUNT = 1

# Market schedule behavior (optimized for Task Scheduler)
RUN_ONLY_MARKET_HOURS = True
EXIT_AFTER_CLOSE = True

# Optional extra logs
LOG_STRIKE_FLOW_TABLE = True
LOG_FULL_CHAIN = False  # huge - keep False

# Error handling
MAX_CONSECUTIVE_ERRORS = 10
RETRY_BACKOFF_SECONDS = 10

# Magnets
MAGNET_COUNT = 3

# Delta approximation
RISK_FREE_RATE = 0.02


# ==========================
# PATHS
# ==========================
EASTERN = ZoneInfo("America/New_York")


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = Path(script_dir())
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

SNAPSHOT_ROOT = DATA_DIR / "snapshots"
CONTRACT_ROOT = DATA_DIR / "contracts"
STRIKEFLOW_ROOT = DATA_DIR / "strike_flow"
FULLCHAIN_ROOT = DATA_DIR / "full_chain"

SNAPSHOT_ROOT.mkdir(exist_ok=True)
CONTRACT_ROOT.mkdir(exist_ok=True)
STRIKEFLOW_ROOT.mkdir(exist_ok=True)
FULLCHAIN_ROOT.mkdir(exist_ok=True)


# ==========================
# TIME HELPERS
# ==========================
def eastern_now():
    return dt.datetime.now(tz=EASTERN)


def is_weekend_et(ts: dt.datetime) -> bool:
    return ts.weekday() >= 5


def market_open_dt(ts: dt.datetime) -> dt.datetime:
    return ts.replace(hour=9, minute=30, second=0, microsecond=0)


def market_close_dt(ts: dt.datetime) -> dt.datetime:
    return ts.replace(hour=16, minute=0, second=0, microsecond=0)


def is_market_open_et(ts: dt.datetime) -> bool:
    if is_weekend_et(ts):
        return False
    return market_open_dt(ts) <= ts < market_close_dt(ts)


def seconds_until_market_open(ts: dt.datetime) -> int:
    """
    Seconds until next market open (9:30 ET).
    If already open -> 0.
    """
    if is_market_open_et(ts):
        return 0

    # weekend -> next Monday open
    if is_weekend_et(ts):
        days_ahead = (7 - ts.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        next_open = (ts + dt.timedelta(days=days_ahead)).replace(
            hour=9, minute=30, second=0, microsecond=0
        )
        return max(0, int((next_open - ts).total_seconds()))

    open_today = market_open_dt(ts)
    close_today = market_close_dt(ts)

    # before open
    if ts < open_today:
        return max(0, int((open_today - ts).total_seconds()))

    # after close -> next weekday open
    next_day = ts + dt.timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += dt.timedelta(days=1)

    next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
    return max(0, int((next_open - ts).total_seconds()))


def minutes_to_market_close(ts: dt.datetime) -> float:
    close = market_close_dt(ts)
    return max(0.0, (close - ts).total_seconds() / 60.0)


def today_et_str(ts: dt.datetime) -> str:
    return ts.strftime("%Y-%m-%d")


def time_to_expiry_years(expiry_str: str, ts_et: dt.datetime) -> float:
    """
    Assume expiration at 4pm ET on expiry date.
    """
    exp_date = dt.datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=EASTERN)
    exp_dt = exp_date.replace(hour=16, minute=0, second=0, microsecond=0)
    secs = (exp_dt - ts_et).total_seconds()
    if secs <= 0:
        return 0.0
    return secs / (365.0 * 24.0 * 3600.0)


# ==========================
# LOGGING
# ==========================
def log_line(msg: str):
    ts = eastern_now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)

    day = eastern_now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"collector_{day}.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ==========================
# FAST PARQUET APPEND (PARTITIONED)
# ==========================
def append_partitioned_parquet(root_dir: Path, df: pd.DataFrame, ts_et: dt.datetime, prefix: str):
    """
    Writes one small parquet file per batch into:
      root_dir / date=YYYY-MM-DD / prefix_YYYYMMDD_HHMMSS_xxxx.parquet
    Fast forever, no rewrites.
    """
    if df is None or df.empty:
        return

    date_str = ts_et.strftime("%Y-%m-%d")
    out_dir = root_dir / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{prefix}_{ts_et.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)}.parquet"
    out_path = out_dir / fname

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path)


# ==========================
# NORMALIZATION
# ==========================
def normalize_chain_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "strike"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    if "lastPrice" in df.columns:
        mask = (df["mid"] <= 0) & (df["lastPrice"] > 0)
        df.loc[mask, "mid"] = df.loc[mask, "lastPrice"]

    for col in ["volume", "openInterest", "mid", "strike", "impliedVolatility", "bid", "ask", "lastPrice"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df[df["strike"] > 0].copy()

    if "contractSymbol" in df.columns:
        df["contractSymbol"] = df["contractSymbol"].astype(str)

    return df


# ==========================
# FETCH
# ==========================
def get_spot_and_volume_1m(ticker_obj: yf.Ticker):
    hist = ticker_obj.history(period="1d", interval="1m")
    if hist.empty:
        raise RuntimeError("Could not fetch SPY intraday 1m history.")
    spot = float(hist["Close"].iloc[-1])
    vol = float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else float("nan")
    return spot, vol


def get_vix_level():
    vix = yf.Ticker(VIX_TICKER)
    hist = vix.history(period="1d", interval="1m")
    if not hist.empty:
        return float(hist["Close"].iloc[-1])

    hist2 = vix.history(period="10d", interval="1d")
    if hist2.empty:
        return float("nan")
    return float(hist2["Close"].iloc[-1])


def pick_0dte_or_next_expiration(expirations: list[str], today_str: str):
    if today_str in expirations:
        return today_str, True
    return expirations[0], False


def fetch_chain_snapshot(ts_et: dt.datetime):
    t = yf.Ticker(TICKER)
    spot, spy_vol_1m = get_spot_and_volume_1m(t)

    expirations = list(t.options)
    if not expirations:
        raise RuntimeError("No expirations returned from Yahoo Finance.")

    exp_used, is_true_0dte = pick_0dte_or_next_expiration(expirations, today_et_str(ts_et))

    chain = t.option_chain(exp_used)
    calls = normalize_chain_df(chain.calls)
    puts = normalize_chain_df(chain.puts)

    return spot, spy_vol_1m, exp_used, is_true_0dte, calls, puts


# ==========================
# FLOW FEATURES
# ==========================
def compute_strike_flow(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, strike_window: float) -> pd.DataFrame:
    c = calls[calls["strike"].between(spot - strike_window, spot + strike_window)].copy()
    p = puts[puts["strike"].between(spot - strike_window, spot + strike_window)].copy()

    c = c[(c["mid"] >= MIN_MID_TO_COUNT) & (c["volume"] >= MIN_VOLUME_TO_COUNT)].copy()
    p = p[(p["mid"] >= MIN_MID_TO_COUNT) & (p["volume"] >= MIN_VOLUME_TO_COUNT)].copy()

    c["calls_premium_proxy"] = c["mid"] * c["volume"] * 100.0
    p["puts_premium_proxy"] = p["mid"] * p["volume"] * 100.0

    c_agg = c.groupby("strike", as_index=False)["calls_premium_proxy"].sum()
    p_agg = p.groupby("strike", as_index=False)["puts_premium_proxy"].sum()

    df = pd.merge(c_agg, p_agg, on="strike", how="outer").fillna(0.0)
    df = df.sort_values("strike").reset_index(drop=True)
    df["net_premium_proxy"] = df["calls_premium_proxy"] - df["puts_premium_proxy"]
    df["total_premium_proxy"] = df["calls_premium_proxy"] + df["puts_premium_proxy"]
    return df


def get_magnets(strike_df: pd.DataFrame, n: int = 3):
    if strike_df is None or strike_df.empty:
        return []
    df = strike_df.sort_values("total_premium_proxy", ascending=False).head(n)
    return df["strike"].astype(float).tolist()


# ==========================
# APPROX DELTA (Black-Scholes)
# ==========================
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, call: bool) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        if call:
            return norm_cdf(d1)
        return norm_cdf(d1) - 1.0
    except Exception:
        return float("nan")


# ==========================
# CONTRACT CANDIDATES
# ==========================
def select_candidate_contracts(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> pd.DataFrame:
    def prep(df: pd.DataFrame, opt_type: str) -> pd.DataFrame:
        d = df.copy()
        d["type"] = opt_type
        d["moneyness"] = d["strike"] / spot
        d["abs_dist"] = (d["strike"] - spot).abs()
        d["spread"] = (d["ask"] - d["bid"]).clip(lower=0.0)

        mid_safe = d["mid"].replace(0, np.nan)
        d["spread_pct_mid"] = d["spread"] / mid_safe

        d = d[d["mid"] >= MIN_MID_TO_COUNT].copy()
        d = d[d["strike"].between(spot - STRIKE_WINDOW_CANDIDATES, spot + STRIKE_WINDOW_CANDIDATES)].copy()

        # safer rank_score
        inv_spread = 1.0 / (d["spread_pct_mid"].fillna(999) + 1e-6)
        inv_dist = 1.0 / (d["abs_dist"] + 1e-6)

        d["rank_score"] = (
            np.log1p(d["volume"]) * 0.45 +
            np.log1p(d["openInterest"]) * 0.35 +
            inv_spread * 0.15 +
            inv_dist * 0.05
        )
        d["rank_score"] = d["rank_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return d

    c = prep(calls, "call").sort_values("rank_score", ascending=False).head(TOP_N_CALLS).copy()
    p = prep(puts, "put").sort_values("rank_score", ascending=False).head(TOP_N_PUTS).copy()

    out = pd.concat([c, p], ignore_index=True)

    keep_cols = [
        "contractSymbol", "type", "strike",
        "bid", "ask", "mid", "lastPrice",
        "volume", "openInterest", "impliedVolatility",
        "moneyness", "abs_dist", "spread", "spread_pct_mid",
        "rank_score"
    ]
    out = out[[c for c in keep_cols if c in out.columns]].copy()
    return out


# ==========================
# SNAPSHOT FEATURE ROW
# ==========================
def build_snapshot_feature_row(ts: dt.datetime, spot: float, spy_vol_1m: float, vix: float,
                              exp_used: str, is_true_0dte: bool,
                              strike_flow_df: pd.DataFrame, magnets: list[float]) -> dict:
    calls_flow = float(strike_flow_df["calls_premium_proxy"].sum()) if not strike_flow_df.empty else 0.0
    puts_flow = float(strike_flow_df["puts_premium_proxy"].sum()) if not strike_flow_df.empty else 0.0
    net_flow = calls_flow - puts_flow
    total_flow = calls_flow + puts_flow
    cp_ratio = calls_flow / (puts_flow + 1e-9)

    magnet_1 = float(magnets[0]) if len(magnets) > 0 else float("nan")
    magnet_2 = float(magnets[1]) if len(magnets) > 1 else float("nan")
    magnet_3 = float(magnets[2]) if len(magnets) > 2 else float("nan")

    dist_to_nearest = float(min([abs(spot - m) for m in magnets])) if magnets else float("nan")

    top_total = float(strike_flow_df["total_premium_proxy"].max()) if not strike_flow_df.empty else 0.0
    pct_top = top_total / (total_flow + 1e-9)

    atm_band = strike_flow_df[strike_flow_df["strike"].between(spot - 2, spot + 2)]
    net_atm = float(atm_band["net_premium_proxy"].sum()) if not atm_band.empty else 0.0

    mins_to_close = minutes_to_market_close(ts)
    T_years = time_to_expiry_years(exp_used, ts)

    return {
        "ts_et": ts,
        "spot": float(spot),
        "spy_volume_1m": float(spy_vol_1m),
        "vix": float(vix),

        "exp_used": str(exp_used),
        "is_true_0dte": bool(is_true_0dte),
        "mins_to_close": float(mins_to_close),
        "T_years_to_expiry": float(T_years),

        "calls_flow": float(calls_flow),
        "puts_flow": float(puts_flow),
        "net_flow": float(net_flow),
        "total_flow": float(total_flow),
        "cp_ratio": float(cp_ratio),

        "magnet_1": magnet_1,
        "magnet_2": magnet_2,
        "magnet_3": magnet_3,
        "dist_to_nearest_magnet": float(dist_to_nearest),

        "top_strike_total_flow": float(top_total),
        "pct_flow_top_strike": float(pct_top),
        "net_flow_atm_band": float(net_atm),

        "n_strikes_in_flow_window": int(len(strike_flow_df)),
        "strike_window_flow": float(STRIKE_WINDOW_FLOW),
    }


# ==========================
# MAIN
# ==========================
def main():
    log_line("=== SPY Contract Collector (Task Scheduler Optimized) ===")
    log_line(f"Folder: {BASE_DIR.resolve()}")
    log_line("")

    consecutive_errors = 0

    while True:
        ts = eastern_now()

        # Market-hours gating (best for Task Scheduler)
        if RUN_ONLY_MARKET_HOURS and not is_market_open_et(ts):
            if EXIT_AFTER_CLOSE and (not is_weekend_et(ts)) and ts >= market_close_dt(ts):
                log_line("Market closed for the day. Exiting collector.")
                break

            wait_s = seconds_until_market_open(ts)
            wait_m = wait_s // 60
            log_line(f"Market closed. Sleeping {wait_m} minutes until next open...")
            time.sleep(max(10, wait_s))
            continue

        try:
            spot, spy_vol_1m, exp_used, is_true_0dte, calls, puts = fetch_chain_snapshot(ts)
            vix_level = get_vix_level()

            strike_flow_df = compute_strike_flow(calls, puts, spot, STRIKE_WINDOW_FLOW)
            magnets = get_magnets(strike_flow_df, n=MAGNET_COUNT)

            snap_row = build_snapshot_feature_row(
                ts=ts,
                spot=spot,
                spy_vol_1m=spy_vol_1m,
                vix=vix_level,
                exp_used=exp_used,
                is_true_0dte=is_true_0dte,
                strike_flow_df=strike_flow_df,
                magnets=magnets
            )
            snap_df = pd.DataFrame([snap_row])
            append_partitioned_parquet(SNAPSHOT_ROOT, snap_df, ts, prefix="snap")

            cand = select_candidate_contracts(calls, puts, spot)
            if not cand.empty:
                cand = cand.copy()
                cand.insert(0, "ts_et", ts)
                cand.insert(1, "exp_used", exp_used)
                cand.insert(2, "spot", float(spot))
                cand.insert(3, "vix", float(vix_level))
                cand.insert(4, "mins_to_close", float(snap_row["mins_to_close"]))
                cand.insert(5, "net_flow", float(snap_row["net_flow"]))
                cand.insert(6, "cp_ratio", float(snap_row["cp_ratio"]))
                cand.insert(7, "magnet_1", float(snap_row["magnet_1"]))
                cand.insert(8, "magnet_2", float(snap_row["magnet_2"]))
                cand.insert(9, "magnet_3", float(snap_row["magnet_3"]))
                cand.insert(10, "dist_to_nearest_magnet", float(snap_row["dist_to_nearest_magnet"]))

                # approx delta
                T_years = float(snap_row["T_years_to_expiry"])
                deltas = []
                for _, r in cand.iterrows():
                    sigma = float(r["impliedVolatility"])
                    call = (str(r["type"]) == "call")
                    d = bs_delta(S=float(spot), K=float(r["strike"]), T=T_years,
                                 r=RISK_FREE_RATE, sigma=sigma, call=call)
                    deltas.append(d)
                cand["delta_approx"] = deltas

                append_partitioned_parquet(CONTRACT_ROOT, cand, ts, prefix="contracts")

            if LOG_STRIKE_FLOW_TABLE and strike_flow_df is not None and not strike_flow_df.empty:
                s = strike_flow_df.copy()
                s.insert(0, "ts_et", ts)
                s.insert(1, "exp_used", exp_used)
                s.insert(2, "spot", float(spot))
                append_partitioned_parquet(STRIKEFLOW_ROOT, s, ts, prefix="strikeflow")

            consecutive_errors = 0
            log_line(
                f"spot={spot:.2f} vix={vix_level:.2f} "
                f"net_flow={snap_row['net_flow']/1e6:.2f}M cand={len(cand)} magnets={magnets}"
            )

        except KeyboardInterrupt:
            log_line("Stopping collector (KeyboardInterrupt).")
            break
        except Exception as e:
            consecutive_errors += 1
            log_line(f"ERROR ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}")
            log_line(traceback.format_exc())

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log_line("Too many errors. Exiting.")
                break

            time.sleep(RETRY_BACKOFF_SECONDS)

        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
