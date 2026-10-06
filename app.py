"""웹 서버: python app.py  →  http://127.0.0.1:5000

옵션:
  --demo   인터넷 없이 합성 데이터로 실행 (동작 확인용)
  --port   포트 (기본 5000)
"""
import argparse
import math

import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from screener import DEFAULT_SECTORS, DataProvider, DemoProvider, Engine, Params

app = Flask(__name__)
engine = Engine(DataProvider())


def clean(o):
    """NaN/inf/numpy 타입을 JSON 안전 값으로 변환."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return None if math.isnan(o) or math.isinf(o) else round(o, 6)
    return o


def ok(data):
    return jsonify(clean(data))


def body():
    return request.get_json(silent=True) or {}


def tickers_of(b):
    raw = b.get("tickers", [])
    if isinstance(raw, str):
        raw = raw.replace(",", " ").split()
    return [t.strip().upper() for t in raw if t.strip()]


@app.errorhandler(Exception)
def on_error(e):
    if isinstance(e, HTTPException):
        return e
    return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.get("/")
def index():
    return render_template("index.html", demo=app.config.get("DEMO", False))


@app.get("/api/defaults")
def defaults():
    from dataclasses import asdict
    return ok({"sectors": DEFAULT_SECTORS, "params": asdict(Params())})


@app.post("/api/sectors/refresh")
def sectors_refresh():
    return ok(engine.refresh_sectors(int(body().get("n", 5))))


@app.post("/api/backtest")
def backtest():
    b = body()
    return ok(engine.backtest(tickers_of(b), b.get("params"), b.get("period", "5y")))


@app.post("/api/optimize")
def optimize():
    b = body()
    return ok(engine.optimize(tickers_of(b), b.get("period", "5y"), b.get("params"),
                              int(b.get("min_signals", 5))))


@app.post("/api/scan")
def scan():
    b = body()
    return ok(engine.scan(b.get("sectors") or DEFAULT_SECTORS, b.get("params"),
                          b.get("period", "5y")))


@app.post("/api/chart")
def chart():
    b = body()
    return ok(engine.chart(b["ticker"], b.get("params"), b.get("period", "5y")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    if a.demo:
        engine.data = DemoProvider()
        app.config["DEMO"] = True
    app.run(host=a.host, port=a.port, debug=False)
