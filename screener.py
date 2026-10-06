"""거래량 증가 + 주가 횡보 패턴 탐지 / 백테스트 / 현재 스캔 엔진.

패턴 정의 (일봉 기준, 기준일 t):
  - 횡보: 최근 `window`일 종가의 (최고-최저)/평균 <= max_range
          그리고 window 시작 대비 종가 변화율 |drift| <= max_drift
  - 거래량 증가: 최근 `window`일 평균 거래량 / 그 이전 `base`일 평균 거래량 >= vol_ratio
백테스트 성공 기준:
  - 신호일 이후 `horizon`거래일 안에 종가 기준 최고 수익률이 target 이상
"""
from __future__ import annotations

import itertools
import math
import threading
import time
import zlib
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 섹터별 시가총액 상위 종목 (기본값). "시총 갱신" 시 후보군에서 실시간 시총으로 상위 5개 재선정.
# ---------------------------------------------------------------------------
SECTOR_CANDIDATES: dict[str, list[str]] = {
    "Technology": ["NVDA", "AAPL", "MSFT", "AVGO", "ORCL", "PLTR", "AMD", "CSCO", "IBM", "CRM", "MU"],
    "Communication Services": ["GOOGL", "META", "NFLX", "TMUS", "DIS", "T", "VZ", "CMCSA"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "BKNG", "TJX", "LOW", "SBUX", "NKE"],
    "Consumer Staples": ["WMT", "COST", "PG", "KO", "PM", "PEP", "MDLZ", "MO"],
    "Financials": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP"],
    "Health Care": ["LLY", "JNJ", "ABBV", "UNH", "MRK", "ABT", "TMO", "ISRG", "AMGN"],
    "Industrials": ["GE", "CAT", "RTX", "UBER", "HON", "BA", "UNP", "ETN", "GEV", "DE"],
    "Energy": ["XOM", "CVX", "COP", "WMB", "EOG", "KMI", "SLB", "EPD", "PSX", "MPC"],
    "Utilities": ["NEE", "SO", "CEG", "DUK", "VST", "AEP", "SRE", "D"],
    "Real Estate": ["PLD", "WELL", "AMT", "EQIX", "SPG", "O", "DLR", "PSA", "CCI"],
    "Materials": ["LIN", "SHW", "NEM", "ECL", "APD", "FCX", "CRH", "CTVA", "NUE"],
}
DEFAULT_SECTORS = {s: c[:5] for s, c in SECTOR_CANDIDATES.items()}


@dataclass
class Params:
    window: int = 20        # 횡보 판정 기간(거래일)
    base: int = 60          # 거래량 비교 기준 기간(window 직전)
    max_range: float = 0.10  # 횡보 박스 폭 상한 (10%)
    max_drift: float = 0.04  # window 동안 순변화율 상한 (±4%)
    vol_ratio: float = 1.3   # 거래량 증가 배수 하한
    horizon: int = 20        # 신호 이후 관찰 기간(거래일)
    target: float = 0.05     # 성공 판정 상승률 (5%)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Params":
        p = cls()
        for k, v in (d or {}).items():
            if hasattr(p, k) and v is not None and v != "":
                setattr(p, k, type(getattr(p, k))(v))
        return p


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------
class DataProvider:
    """yfinance 기반 일봉 데이터. 1시간 메모리 캐시."""

    TTL = 3600

    def __init__(self):
        self._cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        self._lock = threading.Lock()

    def history(self, tickers: list[str], period: str = "5y") -> dict[str, pd.DataFrame]:
        tickers = [t.strip().upper() for t in tickers if t.strip()]
        out, missing = {}, []
        now = time.time()
        with self._lock:
            for t in tickers:
                hit = self._cache.get((t, period))
                if hit and now - hit[0] < self.TTL:
                    out[t] = hit[1]
                else:
                    missing.append(t)
        if missing:
            fetched = self._fetch(missing, period)
            with self._lock:
                for t, df in fetched.items():
                    self._cache[(t, period)] = (now, df)
            out.update(fetched)
        return out

    def _fetch(self, tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                          group_by="ticker", progress=False, threads=True)
        out = {}
        for t in tickers:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df):
                df.index = pd.to_datetime(df.index).tz_localize(None)
                out[t] = df
        return out

    def market_caps(self, tickers: list[str]) -> dict[str, float]:
        import yfinance as yf

        caps = {}
        for t in tickers:
            try:
                caps[t] = float(yf.Ticker(t).fast_info["market_cap"] or 0)
            except Exception:
                caps[t] = 0.0
        return caps


class DemoProvider(DataProvider):
    """네트워크 없이 동작 확인용 합성 데이터 (횡보+거래량 증가 후 상승 패턴을 심어 둠)."""

    def _fetch(self, tickers, period):
        years = {"1y": 1, "2y": 2, "3y": 3, "5y": 5, "10y": 10, "max": 10}.get(period, 5)
        n = 252 * years
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
        out = {}
        for t in tickers:
            rng = np.random.default_rng(zlib.crc32(t.encode()))
            rets = rng.normal(0.0004, 0.017, n)
            vol = rng.lognormal(16, 0.25, n)
            i = 80
            while i < n - 60:
                if rng.random() < 0.5:
                    L = int(rng.integers(20, 35))
                    rets[i:i + L] = rng.normal(0, 0.006, L)
                    rets[i:i + L] -= rets[i:i + L].mean()
                    vol[i:i + L] *= rng.uniform(1.4, 2.0)
                    if rng.random() < 0.65:
                        rets[i + L:i + L + 15] += rng.uniform(0.004, 0.01)
                    i += L + 40
                else:
                    i += int(rng.integers(30, 90))
            close = 100 * np.exp(np.cumsum(rets))
            out[t] = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                                   "Close": close, "Volume": vol.round()}, index=idx)
        return out

    def market_caps(self, tickers):
        return {t: float(zlib.crc32(t.encode()) % 3_000_000) * 1e6 for t in tickers}


# ---------------------------------------------------------------------------
# 지표 / 신호
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    close, vol = df["Close"], df["Volume"]
    w = p.window
    hi, lo = close.rolling(w).max(), close.rolling(w).min()
    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out["volume"] = vol
    out["box_high"] = hi
    out["box_low"] = lo
    out["range_pct"] = (hi - lo) / close.rolling(w).mean()
    out["drift"] = close / close.shift(w - 1) - 1
    out["vol_ratio"] = vol.rolling(w).mean() / vol.shift(w).rolling(p.base).mean()
    out["signal"] = ((out["range_pct"] <= p.max_range)
                     & (out["drift"].abs() <= p.max_drift)
                     & (out["vol_ratio"] >= p.vol_ratio))
    # 향후 horizon일 성과 (t+1 ~ t+h)
    h = p.horizon
    out["fwd_ret"] = close.shift(-h) / close - 1
    out["fwd_max"] = close.shift(-1).rolling(h).max().shift(-(h - 1)) / close - 1
    out["fwd_min"] = close.shift(-1).rolling(h).min().shift(-(h - 1)) / close - 1
    return out


def extract_events(ind: pd.DataFrame, p: Params) -> list[dict]:
    """연속된 신호는 첫날만 진입으로 보고, 이후 horizon일 동안은 중복 신호 무시."""
    events, last = [], -10**9
    sig_pos = np.flatnonzero(ind["signal"].to_numpy())
    for i in sig_pos:
        if i - last < p.horizon:
            continue
        last = i
        r = ind.iloc[i]
        done = not math.isnan(r["fwd_max"])
        events.append({
            "date": ind.index[i].strftime("%Y-%m-%d"),
            "start": ind.index[max(0, i - p.window + 1)].strftime("%Y-%m-%d"),
            "close": r["close"],
            "range_pct": r["range_pct"],
            "drift": r["drift"],
            "vol_ratio": r["vol_ratio"],
            "fwd_ret": r["fwd_ret"] if done else None,
            "fwd_max": r["fwd_max"] if done else None,
            "fwd_min": r["fwd_min"] if done else None,
            "breakout": bool(r["fwd_max"] + 1 > r["box_high"] / r["close"]) if done else None,
            "success": bool(r["fwd_max"] >= p.target) if done else None,
            "pending": not done,
        })
    return events


def summarize(events: list[dict], ind: pd.DataFrame | None, p: Params) -> dict:
    done = [e for e in events if not e["pending"]]
    n = len(done)
    s = {
        "signals": n,
        "pending": len(events) - n,
        "wins": sum(e["success"] for e in done),
        "win_rate": (sum(e["success"] for e in done) / n) if n else None,
        "avg_fwd_ret": float(np.mean([e["fwd_ret"] for e in done])) if n else None,
        "avg_fwd_max": float(np.mean([e["fwd_max"] for e in done])) if n else None,
        "avg_fwd_min": float(np.mean([e["fwd_min"] for e in done])) if n else None,
    }
    if ind is not None:
        valid = ind["fwd_max"].dropna()
        s["base_win_rate"] = float((valid >= p.target).mean()) if len(valid) else None
        s["base_fwd_ret"] = float(ind["fwd_ret"].dropna().mean()) if len(valid) else None
        if s["win_rate"] is not None and s["base_win_rate"]:
            s["edge"] = s["win_rate"] - s["base_win_rate"]
    return s


# ---------------------------------------------------------------------------
# 상위 API
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, provider: DataProvider):
        self.data = provider

    def backtest(self, tickers: list[str], params: dict | None, period="5y") -> dict:
        p = Params.from_dict(params)
        hist = self.data.history(tickers, period)
        results, all_events = [], []
        for t in tickers:
            t = t.strip().upper()
            df = hist.get(t)
            if df is None or len(df) < p.window + p.base + 5:
                results.append({"ticker": t, "error": "데이터 없음/부족"})
                continue
            ind = compute_indicators(df, p)
            ev = extract_events(ind, p)
            for e in ev:
                e["ticker"] = t
            all_events += ev
            results.append({"ticker": t, **summarize(ev, ind, p), "events": ev})
        ok = [r for r in results if "error" not in r]
        # 선별: 신호 2회 이상 & 승률이 무조건 기준 대비 높은 종목
        selected = sorted(
            [r for r in ok if r["signals"] >= 2 and r["win_rate"] is not None
             and r["win_rate"] >= max(0.5, r.get("base_win_rate") or 0)],
            key=lambda r: (r["win_rate"], r["avg_fwd_max"] or 0), reverse=True)
        return {
            "params": asdict(p),
            "overall": summarize(all_events, None, p),
            "results": results,
            "selected": [r["ticker"] for r in selected],
            "winners": sorted([e for e in all_events if e["success"]],
                              key=lambda e: e["date"], reverse=True),
        }

    def optimize(self, tickers: list[str], period="5y", base_params: dict | None = None,
                 min_signals: int = 5) -> list[dict]:
        bp = Params.from_dict(base_params)
        hist = self.data.history(tickers, period)
        grid = itertools.product([10, 15, 20, 30], [0.06, 0.08, 0.10, 0.12],
                                 [1.2, 1.3, 1.5, 1.8])
        rows = []
        for w, rg, vr in grid:
            p = Params(**{**asdict(bp), "window": w, "max_range": rg, "vol_ratio": vr,
                          "max_drift": min(bp.max_drift, rg / 2)})
            evs, base_hits, base_n = [], 0, 0
            for t, df in hist.items():
                if len(df) < p.window + p.base + 5:
                    continue
                ind = compute_indicators(df, p)
                evs += extract_events(ind, p)
                v = ind["fwd_max"].dropna()
                base_hits += int((v >= p.target).sum())
                base_n += len(v)
            s = summarize(evs, None, p)
            if s["signals"] < min_signals:
                continue
            s["base_win_rate"] = base_hits / base_n if base_n else None
            s["edge"] = s["win_rate"] - (s["base_win_rate"] or 0)
            rows.append({"params": asdict(p), **s})
        # 표본 수를 감안한 점수 (승률 하한 근사: Wilson lower bound)
        for r in rows:
            r["score"] = wilson_lower(r["wins"], r["signals"])
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:15]

    def scan(self, sectors: dict[str, list[str]], params: dict | None, period="5y") -> dict:
        p = Params.from_dict(params)
        tickers = sorted({t.upper() for ts in sectors.values() for t in ts})
        hist = self.data.history(tickers, period)
        rows = []
        for sector, ts in sectors.items():
            for t in ts:
                t = t.upper()
                df = hist.get(t)
                if df is None or len(df) < p.window + p.base + 5:
                    rows.append({"sector": sector, "ticker": t, "error": "데이터 없음/부족"})
                    continue
                ind = compute_indicators(df, p)
                last = ind.iloc[-1]
                streak = 0
                for v in ind["signal"].to_numpy()[::-1]:
                    if not v:
                        break
                    streak += 1
                hist_sum = summarize(extract_events(ind, p), ind, p)
                rows.append({
                    "sector": sector, "ticker": t,
                    "date": ind.index[-1].strftime("%Y-%m-%d"),
                    "close": last["close"],
                    "range_pct": last["range_pct"], "drift": last["drift"],
                    "vol_ratio": last["vol_ratio"],
                    "match": bool(last["signal"]), "streak": streak,
                    "near": near_score(last, p),
                    "hist_signals": hist_sum["signals"], "hist_win_rate": hist_sum["win_rate"],
                    "hist_avg_fwd_max": hist_sum["avg_fwd_max"],
                    "base_win_rate": hist_sum.get("base_win_rate"),
                })
        return {"params": asdict(p), "rows": rows,
                "matches": [r["ticker"] for r in rows if r.get("match")]}

    def chart(self, ticker: str, params: dict | None, period="5y") -> dict:
        p = Params.from_dict(params)
        df = self.data.history([ticker], period).get(ticker.upper())
        if df is None:
            return {"error": "데이터 없음"}
        ind = compute_indicators(df, p)
        return {
            "ticker": ticker.upper(),
            "dates": [d.strftime("%Y-%m-%d") for d in ind.index],
            "close": ind["close"].round(4).tolist(),
            "volume": ind["volume"].tolist(),
            "signal": ind["signal"].tolist(),
            "events": extract_events(ind, p),
        }

    def refresh_sectors(self, n: int = 5) -> dict:
        all_t = [t for ts in SECTOR_CANDIDATES.values() for t in ts]
        caps = self.data.market_caps(all_t)
        out = {}
        for s, ts in SECTOR_CANDIDATES.items():
            ranked = sorted(ts, key=lambda t: caps.get(t, 0), reverse=True)
            out[s] = [{"ticker": t, "market_cap": caps.get(t, 0)} for t in ranked[:n]]
        return out


def near_score(r, p: Params) -> float:
    """조건 충족 정도 0~1 (1이면 세 조건 모두 충족)."""
    if any(pd.isna(r[k]) for k in ("range_pct", "drift", "vol_ratio")):
        return 0.0
    a = min(1.0, p.max_range / r["range_pct"]) if r["range_pct"] > 0 else 1.0
    b = min(1.0, p.max_drift / abs(r["drift"])) if r["drift"] != 0 else 1.0
    c = min(1.0, r["vol_ratio"] / p.vol_ratio)
    return round((a + b + c) / 3, 3)


def wilson_lower(k: int, n: int, z: float = 1.64) -> float:
    if n == 0:
        return 0.0
    ph = k / n
    return (ph + z * z / (2 * n) - z * math.sqrt((ph * (1 - ph) + z * z / (4 * n)) / n)) / (1 + z * z / n)
