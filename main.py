from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import yfinance as yf
<<<<<<< HEAD
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QKeySequence
=======
from PySide6.QtCore import QTimer, Qt
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


SYMBOL_GROUPS: dict[str, list[str]] = {
    "Taiwan ETFs": ["0050.TW", "0056.TW", "006208.TW", "00878.TW", "00919.TW"],
    "US ETFs": ["SPY", "QQQ", "TQQQ", "SOXX", "SOXL", "VOO", "VTI"],
    "US Large Caps": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AVGO", "BRK.B",
        "JPM", "V", "MA", "LLY", "UNH", "XOM", "WMT", "COST", "NFLX", "AMD", "ORCL",
    ],
    "Crypto": ["BTC/USDT", "ETH/USDT"],
}

TRADING_FEE_RATE = 0.001  # 0.1%
<<<<<<< HEAD
EMA_WINDOW = 200
EMA_LOOKBACK_DAYS = 400

STRATEGY_PERIODIC_ALL_IN = "periodic_all_in"
STRATEGY_EMA200_ACCUMULATE = "ema200_accumulate"
STRATEGY_LABELS: dict[str, str] = {
    STRATEGY_PERIODIC_ALL_IN: "Periodic Buy (Use available cash every cycle)",
    STRATEGY_EMA200_ACCUMULATE: "EMA200 Trigger (Accumulate cash until price <= EMA200)",
}
=======
DIVIDEND_MODE_CASH = "cash"
DIVIDEND_MODE_REINVEST = "reinvest"
DIVIDEND_MODE_LABELS = {
    DIVIDEND_MODE_CASH: "Cash Dividends",
    DIVIDEND_MODE_REINVEST: "Auto Reinvest Dividends",
}
DISPLAY_MODE_ORIGINAL = "original"
DISPLAY_MODE_NTD = "ntd"
DISPLAY_MODE_LABELS = {
    DISPLAY_MODE_ORIGINAL: "Original Currency",
    DISPLAY_MODE_NTD: "Convert All To NTD",
}
STOCK_CLOSE_CACHE: dict[tuple[str, pd.Timestamp, pd.Timestamp], pd.Series] = {}
STOCK_DIVIDEND_CACHE: dict[tuple[str, pd.Timestamp, pd.Timestamp], pd.Series] = {}
CRYPTO_HISTORY_CACHE: dict[tuple[str, pd.Timestamp, pd.Timestamp], AssetHistory] = {}
FX_HISTORY_CACHE: dict[tuple[pd.Timestamp, pd.Timestamp], pd.Series] = {}
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)


@dataclass
class BacktestResult:
    symbol: str
    strategy_label: str
    invested: float
    position_value: float
    final_value: float
    profit: float
    return_pct: float
    annualized_pct: float
    periods_executed: int
    periods_planned: int
    currency: str
<<<<<<< HEAD
    cash_remaining: float
=======
    invested_ntd: float
    position_value_ntd: float
    final_value_ntd: float
    profit_ntd: float
    return_pct_ntd: float
    annualized_pct_ntd: float
    dividend_cash: float
    dividends_received: float
    reinvested_units: float
    valuation_fx_rate: float | None
    valuation_fx_date: pd.Timestamp | None
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)
    trades: list["TradeRecord"]


@dataclass
class TradeRecord:
    planned_dates: list[pd.Timestamp]
    trade_date: pd.Timestamp
    price: float
    units: float
    spent: float
    fee: float


@dataclass
<<<<<<< HEAD
class AlignedTradeDay:
    planned_dates: list[pd.Timestamp]
    trade_date: pd.Timestamp


class CopyableTableWidget(QTableWidget):
    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection()
            event.accept()
            return
        super().keyPressEvent(event)

    def copy_selection(self) -> None:
        indexes = self.selectedIndexes()
        if not indexes:
            return

        indexes = sorted(indexes, key=lambda idx: (idx.row(), idx.column()))
        rows: dict[int, dict[int, str]] = {}
        selected_columns = sorted({idx.column() for idx in indexes})

        for idx in indexes:
            rows.setdefault(idx.row(), {})[idx.column()] = idx.data() or ""

        lines: list[str] = []
        for row_no in sorted(rows):
            row_values = [rows[row_no].get(col, "") for col in selected_columns]
            lines.append("\t".join(row_values))

        QGuiApplication.clipboard().setText("\n".join(lines))
=======
class AssetHistory:
    close: pd.Series
    dividends: pd.Series


class SortableTableWidgetItem(QTableWidgetItem):
    def __init__(self, text: str, sort_value: object) -> None:
        super().__init__(text)
        self.sort_value = sort_value

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, SortableTableWidgetItem):
            return self.sort_value < other.sort_value
        return super().__lt__(other)
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)


def is_crypto_symbol(symbol: str) -> bool:
    return "/" in symbol


def is_us_symbol(symbol: str) -> bool:
    if is_crypto_symbol(symbol):
        return True
    return not symbol.endswith(".TW")


def to_yf_symbol(symbol: str) -> str:
    # Yahoo Finance uses dash for class shares (e.g., BRK-B).
    if symbol.upper() == "BRK.B":
        return "BRK-B"
    return symbol


def to_exchange_candidates(symbol: str) -> list[tuple[str, str]]:
    base = symbol.split("/")[0].upper()
    requested_quote = symbol.split("/")[1].upper() if "/" in symbol else "USDT"

    candidates: list[tuple[str, str]] = []
    # Prefer USDT pairs on Binance; if unavailable, fallback to USD pairs.
    for market in (f"{base}/{requested_quote}", f"{base}/USD", f"{base}/USDT"):
        if ("binance", market) not in candidates:
            candidates.append(("binance", market))
    for ex_id, market in (("kraken", f"{base}/USD"), ("coinbase", f"{base}/USD")):
        if (ex_id, market) not in candidates:
            candidates.append((ex_id, market))
    return candidates


def normalize_range(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()


def stock_cache_key(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    norm_start, norm_end = normalize_range(start, end)
    return symbol, norm_start, norm_end


def fx_cache_key(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    return normalize_range(start, end)


def normalize_history_index(index: pd.Index) -> pd.DatetimeIndex:
    normalized = pd.DatetimeIndex(pd.to_datetime(index))
    if normalized.tz is not None:
        normalized = normalized.tz_localize(None)
    return normalized.normalize()


def fetch_crypto_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> AssetHistory:
    cache_key = stock_cache_key(symbol, start, end)
    if cache_key in CRYPTO_HISTORY_CACHE:
        return CRYPTO_HISTORY_CACHE[cache_key]

    since = int((start - pd.Timedelta(days=10)).timestamp() * 1000)
    end_ms = int((end + pd.Timedelta(days=1)).timestamp() * 1000)
    errors: list[str] = []

    for exchange_id, market in to_exchange_candidates(symbol):
        try:
            ex_class = getattr(ccxt, exchange_id)
            exchange = ex_class({"enableRateLimit": True})
            exchange.load_markets()
            if market not in exchange.markets:
                errors.append(f"{exchange_id}: {market} not available")
                continue

            rows: list[list[float]] = []
            cursor = since
            while cursor < end_ms:
                batch = exchange.fetch_ohlcv(market, timeframe="1d", since=cursor, limit=1000)
                if not batch:
                    break
                rows.extend(batch)
                next_cursor = int(batch[-1][0]) + 86_400_000
                if next_cursor <= cursor:
                    break
                cursor = next_cursor

            if not rows:
                errors.append(f"{exchange_id}: no OHLCV data")
                continue

            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df = df.drop_duplicates(subset="ts", keep="last")
            dt = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
            close = pd.Series(df["close"].astype(float).values, index=dt, name="close").sort_index()
            close = close[(close.index >= start) & (close.index <= end)].dropna()
            if close.empty:
                errors.append(f"{exchange_id}: no data in selected range")
                continue
            history = AssetHistory(close=close, dividends=pd.Series(dtype=float, name="dividends"))
            CRYPTO_HISTORY_CACHE[cache_key] = history
            return history
        except Exception as exc:
            errors.append(f"{exchange_id}: {exc}")

    err = "; ".join(errors) if errors else "unknown error"
    raise ValueError(f"{symbol}: Unable to fetch crypto history via ccxt ({err}).")


def extract_close_from_download(df: pd.DataFrame, yf_symbol: str) -> pd.Series:
    if df.empty:
        raise ValueError(f"{yf_symbol}: No historical data found.")

    if isinstance(df.columns, pd.MultiIndex):
        close_key = ("Close", yf_symbol)
        if close_key not in df.columns:
            raise ValueError(f"{yf_symbol}: Close price not found in batch response.")
        close = df[close_key]
    else:
        close = df["Close"]

    close = close.dropna()
    close.index = normalize_history_index(close.index)
    return close.sort_index()


def download_stock_close_histories(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.Series]:
    if not symbols:
        return {}

    yf_symbols = [to_yf_symbol(symbol) for symbol in symbols]
    df = yf.download(
        yf_symbols if len(yf_symbols) > 1 else yf_symbols[0],
        start=(start - pd.Timedelta(days=10)).date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        auto_adjust=False,
        progress=False,
        actions=False,
    )
    if df.empty:
        return {}

    closes: dict[str, pd.Series] = {}
    for symbol in symbols:
        yf_symbol = to_yf_symbol(symbol)
        try:
            close = extract_close_from_download(df, yf_symbol)
        except Exception:
            continue
        close = close[(close.index >= start) & (close.index <= end)]
        if not close.empty:
            closes[symbol] = close
    return closes


def prefetch_stock_close_histories(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> None:
    unique_symbols = sorted({symbol for symbol in symbols if not is_crypto_symbol(symbol)})
    uncached = [symbol for symbol in unique_symbols if stock_cache_key(symbol, start, end) not in STOCK_CLOSE_CACHE]
    if not uncached:
        return

    for symbol, close in download_stock_close_histories(uncached, start, end).items():
        STOCK_CLOSE_CACHE[stock_cache_key(symbol, start, end)] = close


def download_stock_dividend_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    yf_symbol = to_yf_symbol(symbol)
    dividends = yf.Ticker(yf_symbol).dividends
    if dividends is None or dividends.empty:
        return pd.Series(dtype=float, name="dividends")

    dividend_series = dividends.astype(float).copy()
    dividend_series.index = normalize_history_index(dividend_series.index)
    dividend_series = dividend_series.groupby(dividend_series.index).sum().sort_index()
    return dividend_series[
        (dividend_series.index >= start.normalize()) & (dividend_series.index <= end.normalize())
    ]


def prefetch_stock_dividend_histories(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    unique_symbols = sorted({symbol for symbol in symbols if not is_crypto_symbol(symbol)})
    uncached = [symbol for symbol in unique_symbols if stock_cache_key(symbol, start, end) not in STOCK_DIVIDEND_CACHE]
    if not uncached:
        return []

    errors: list[str] = []
    max_workers = min(8, len(uncached))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_stock_dividend_history, symbol, start, end): symbol
            for symbol in uncached
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                STOCK_DIVIDEND_CACHE[stock_cache_key(symbol, start, end)] = future.result()
            except Exception as exc:
                errors.append(f"{symbol}: {exc}")
    return errors


def fetch_stock_dividend_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    cache_key = stock_cache_key(symbol, start, end)
    if cache_key in STOCK_DIVIDEND_CACHE:
        return STOCK_DIVIDEND_CACHE[cache_key]

    dividend_series = download_stock_dividend_history(symbol, start, end)
    STOCK_DIVIDEND_CACHE[cache_key] = dividend_series
    return dividend_series


def fetch_stock_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> AssetHistory:
    cache_key = stock_cache_key(symbol, start, end)
    if cache_key not in STOCK_CLOSE_CACHE:
        prefetch_stock_close_histories([symbol], start, end)
    if cache_key not in STOCK_CLOSE_CACHE:
        downloaded = download_stock_close_histories([symbol], start, end)
        if symbol in downloaded:
            STOCK_CLOSE_CACHE[cache_key] = downloaded[symbol]

    close = STOCK_CLOSE_CACHE.get(cache_key)
    if close is None or close.empty:
        raise ValueError(f"{symbol}: No trading data found in selected range.")

    dividends = fetch_stock_dividend_history(symbol, start, end)
    return AssetHistory(close=close, dividends=dividends)


def fetch_usd_twd_history(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    cache_key = fx_cache_key(start, end)
    if cache_key in FX_HISTORY_CACHE:
        return FX_HISTORY_CACHE[cache_key]

    df = yf.download(
        "TWD=X",
        start=(start - pd.Timedelta(days=30)).date().isoformat(),
        end=(end + pd.Timedelta(days=30)).date().isoformat(),
        auto_adjust=False,
        progress=False,
        actions=False,
    )
    if df.empty:
        raise ValueError("Unable to fetch historical USD/TWD exchange rates.")

    if isinstance(df.columns, pd.MultiIndex):
        close = df[("Close", "TWD=X")] if ("Close", "TWD=X") in df.columns else df[("Adj Close", "TWD=X")]
    else:
        close = df["Close"] if "Close" in df.columns else df["Adj Close"]

    close = close.dropna()
    close.index = normalize_history_index(close.index)
    if close.empty:
        raise ValueError("Historical USD/TWD exchange rate data is empty.")
    history = close.sort_index()
    FX_HISTORY_CACHE[cache_key] = history
    return history


def fetch_usd_twd_rate() -> float:
    df = yf.download("TWD=X", period="7d", interval="1d", auto_adjust=False, progress=False, actions=False)
    if df.empty:
        raise ValueError("Unable to fetch USD/TWD exchange rate.")

    if isinstance(df.columns, pd.MultiIndex):
        close = df[("Close", "TWD=X")] if ("Close", "TWD=X") in df.columns else df[("Adj Close", "TWD=X")]
    else:
        close = df["Close"] if "Close" in df.columns else df["Adj Close"]

    close = close.dropna()
    if close.empty:
        raise ValueError("USD/TWD exchange rate data is empty.")

    rate = float(close.iloc[-1])
    if rate <= 0:
        raise ValueError("Invalid USD/TWD exchange rate.")
    return rate


def lookup_history_value(
    series: pd.Series,
    target_date: pd.Timestamp,
    *,
    prefer_on_or_before: bool = True,
) -> tuple[float, pd.Timestamp]:
    if series.empty:
        raise ValueError("No history available for lookup.")

    normalized_target = pd.Timestamp(target_date).normalize()
    index = series.index
    pos = index.searchsorted(normalized_target)

    if pos < len(index) and index[pos] == normalized_target:
        matched_date = pd.Timestamp(index[pos])
    elif prefer_on_or_before and pos > 0:
        matched_date = pd.Timestamp(index[pos - 1])
    elif pos < len(index):
        matched_date = pd.Timestamp(index[pos])
    else:
        matched_date = pd.Timestamp(index[-1])

    value = float(series.loc[matched_date])
    if value <= 0:
        raise ValueError(f"Invalid history value on {matched_date.date()}.")
    return value, matched_date


def generate_schedule(start: pd.Timestamp, end: pd.Timestamp, period_days: int) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    cur = pd.Timestamp(start)
    while cur <= end:
        dates.append(cur)
        cur = cur + pd.Timedelta(days=period_days)
    return dates


def nearest_trade_days(index: pd.DatetimeIndex, scheduled: list[pd.Timestamp]) -> list[pd.Timestamp]:
    aligned: list[pd.Timestamp] = []
    for dt in scheduled:
        pos = index.searchsorted(dt)
        if pos < len(index):
            aligned.append(index[pos])
    return sorted(set(aligned))


def fetch_asset_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> AssetHistory:
    if is_crypto_symbol(symbol):
        return fetch_crypto_history(symbol, start, end)
    return fetch_stock_history(symbol, start, end)


def format_units(symbol: str, units: float) -> str:
    if is_crypto_symbol(symbol):
        return f"{units:.8f}"
    return f"{units:.1f}"


def format_reinvested_units(symbol: str, units: float) -> str:
    if is_crypto_symbol(symbol):
        return f"{units:.8f}"
    return f"{units:.4f}"


def validate_history_start(symbol: str, start: pd.Timestamp, close: pd.Series, grace_days: int = 40) -> None:
    if close.empty:
        return
    first = pd.Timestamp(close.index[0])
    if first > (start + pd.Timedelta(days=grace_days)):
        raise ValueError(
            f"{symbol}: Start date is {start.date()}, but earliest available data is {first.date()}. "
            "Please move the backtest start date forward."
        )


<<<<<<< HEAD
def contribution_in_quote(symbol: str, contribution_ntd: float, usd_twd_rate: float) -> float:
    return contribution_ntd if not is_us_symbol(symbol) else (contribution_ntd / usd_twd_rate)


def build_aligned_schedule(
    index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    period_days: int,
) -> tuple[list[pd.Timestamp], list[AlignedTradeDay]]:
    scheduled = generate_schedule(start, end, period_days)
    grouped_schedule: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for planned_dt in scheduled:
        pos = index.searchsorted(planned_dt)
        if pos < len(index):
            trade_dt = pd.Timestamp(index[pos])
            grouped_schedule.setdefault(trade_dt, []).append(planned_dt)

    aligned_schedule = [
        AlignedTradeDay(planned_dates=planned_dates, trade_date=trade_dt)
        for trade_dt, planned_dates in sorted(grouped_schedule.items())
    ]
    return scheduled, aligned_schedule


def format_planned_dates(planned_dates: list[pd.Timestamp]) -> str:
    if not planned_dates:
        return "-"

    first = planned_dates[0].date().isoformat()
    if len(planned_dates) == 1:
        return first

    last = planned_dates[-1].date().isoformat()
    return f"{first} to {last} ({len(planned_dates)} periods)"


def calc_order_from_budget(symbol: str, price: float, budget: float) -> tuple[float, float, float]:
    if price <= 0 or budget <= 0:
        return 0.0, 0.0, 0.0

    max_spent_before_fee = budget / (1.0 + TRADING_FEE_RATE)
    if max_spent_before_fee <= 0:
        return 0.0, 0.0, 0.0

    if is_crypto_symbol(symbol):
        units = max_spent_before_fee / price
    else:
        units = np.floor((max_spent_before_fee / price) * 10) / 10

    if units <= 0:
        return 0.0, 0.0, 0.0

    spent = units * price
    fee = spent * TRADING_FEE_RATE
    total_cost = spent + fee
    if total_cost > budget + 1e-9:
        return 0.0, 0.0, 0.0
    return units, spent, fee


def finalize_backtest_result(
    symbol: str,
    strategy_label: str,
    close: pd.Series,
    total_contributed: float,
    shares: float,
    cash: float,
    executed_periods: int,
    scheduled_count: int,
    trade_records: list[TradeRecord],
) -> BacktestResult:
    if total_contributed <= 0:
        raise ValueError(f"{symbol}: No capital was contributed in the selected period.")

    last_price = float(close.iloc[-1])
    final_value = shares * last_price + cash
    profit = final_value - total_contributed
    ret = (final_value / total_contributed - 1.0) if total_contributed > 0 else 0.0

    days = max((close.index[-1] - close.index[0]).days, 1)
    years = days / 365.25
    annualized = ((final_value / total_contributed) ** (1 / years) - 1.0) if total_contributed > 0 and years > 0 else 0.0

    return BacktestResult(
        symbol=symbol,
        strategy_label=strategy_label,
        invested=total_contributed,
        final_value=final_value,
        profit=profit,
        return_pct=ret * 100,
        annualized_pct=annualized * 100,
        periods_executed=executed_periods,
        periods_planned=scheduled_count,
        currency="USDT" if is_crypto_symbol(symbol) else ("USD" if is_us_symbol(symbol) else "TWD"),
        cash_remaining=cash,
        trades=trade_records,
    )


def run_periodic_all_in_backtest(
=======
def align_to_trading_day(index: pd.DatetimeIndex, target_date: pd.Timestamp) -> pd.Timestamp | None:
    pos = index.searchsorted(target_date)
    if pos < len(index):
        return pd.Timestamp(index[pos])
    return None


def run_dca_backtest(
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    period_days: int,
    contribution_ntd: float,
    usd_twd_history: pd.Series | None,
    dividend_mode: str,
) -> BacktestResult:
    history = fetch_asset_history(symbol, start, end)
    close = history.close
    dividends = history.dividends
    requires_fx = is_us_symbol(symbol)

    if close.empty:
        raise ValueError(f"{symbol}: No trading data in the selected date range.")
    if requires_fx and usd_twd_history is None:
        raise ValueError(f"{symbol}: Historical USD/TWD exchange rates are required.")
    validate_history_start(symbol, start, close)

    scheduled, aligned_schedule = build_aligned_schedule(close.index, start, end, period_days)
    if not aligned_schedule:
        raise ValueError(f"{symbol}: Unable to align any DCA dates to trading days.")

    cash = 0.0
    shares = 0.0
<<<<<<< HEAD
    total_contributed = 0.0
=======
    invested = 0.0
    invested_ntd = 0.0
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)
    executed_periods = 0
    dividend_cash = 0.0
    dividends_received = 0.0
    reinvested_units = 0.0
    trade_records: list[TradeRecord] = []
    contribution_quote = contribution_in_quote(symbol, contribution_ntd, usd_twd_rate)
    pending_planned_dates: list[pd.Timestamp] = []

<<<<<<< HEAD
    for aligned_day in aligned_schedule:
        dt = aligned_day.trade_date
        price = float(close.loc[dt])
        contribution_total = contribution_quote * len(aligned_day.planned_dates)
        cash += contribution_total
        total_contributed += contribution_total
        pending_planned_dates.extend(aligned_day.planned_dates)

        units, spent, fee = calc_order_from_budget(symbol, price, cash)
        if units <= 0:
            continue

        shares += units
        cash -= spent + fee
        executed_periods += len(pending_planned_dates)
        trade_records.append(
            TradeRecord(
                planned_dates=list(pending_planned_dates),
                trade_date=dt,
                price=price,
                units=units,
                spent=spent,
                fee=fee,
            )
        )
        pending_planned_dates.clear()
=======
    trades_by_date: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for planned_dt, trade_dt in aligned_schedule:
        trades_by_date.setdefault(pd.Timestamp(trade_dt), []).append(planned_dt)

    dividends_by_date: dict[pd.Timestamp, float] = {}
    if not dividends.empty:
        for dividend_dt, cash_per_share in dividends.items():
            aligned_dividend_dt = align_to_trading_day(close.index, pd.Timestamp(dividend_dt))
            if aligned_dividend_dt is None:
                continue
            dividends_by_date[aligned_dividend_dt] = (
                dividends_by_date.get(aligned_dividend_dt, 0.0) + float(cash_per_share)
            )

    event_dates = sorted(set(trades_by_date) | set(dividends_by_date))
    for dt in event_dates:
        dividend_per_share = dividends_by_date.get(dt, 0.0)
        if dividend_per_share > 0 and shares > 0:
            cash_received = shares * dividend_per_share
            dividends_received += cash_received
            if dividend_mode == DIVIDEND_MODE_REINVEST:
                price = float(close.loc[dt])
                if price > 0:
                    extra_units = cash_received / price
                    shares += extra_units
                    reinvested_units += extra_units
                else:
                    dividend_cash += cash_received
            else:
                dividend_cash += cash_received

        for planned_dt in trades_by_date.get(dt, []):
            price = float(close.loc[dt])
            if price <= 0:
                continue
            trade_fx_rate = 1.0
            if requires_fx:
                assert usd_twd_history is not None
                trade_fx_rate, _ = lookup_history_value(usd_twd_history, dt)
            contribution_in_quote = contribution_ntd if not requires_fx else (contribution_ntd / trade_fx_rate)
            if is_crypto_symbol(symbol):
                units = contribution_in_quote / price
            else:
                units = np.floor((contribution_in_quote / price) * 10) / 10
            if units <= 0:
                continue
            spent = units * price
            fee = spent * TRADING_FEE_RATE
            shares += units
            invested += spent + fee
            invested_ntd += (spent + fee) * trade_fx_rate
            executed_periods += 1
            trade_records.append(
                TradeRecord(
                    planned_date=planned_dt,
                    trade_date=dt,
                    price=price,
                    units=units,
                    spent=spent,
                    fee=fee,
                )
            )
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)

    if shares <= 0 and cash <= 0:
        if is_crypto_symbol(symbol):
            raise ValueError(f"{symbol}: Amount (NTD) is too low to buy crypto in selected periods.")
        raise ValueError(f"{symbol}: Amount (NTD) is too low to buy at least 0.1 share in selected periods.")

<<<<<<< HEAD
    return finalize_backtest_result(
        symbol=symbol,
        strategy_label=STRATEGY_LABELS[STRATEGY_PERIODIC_ALL_IN],
        close=close,
        total_contributed=total_contributed,
        shares=shares,
        cash=cash,
        executed_periods=executed_periods,
        scheduled_count=len(scheduled),
        trade_records=trade_records,
=======
    last_price = float(close.iloc[-1])
    position_value = shares * last_price
    final_value = position_value + dividend_cash
    profit = final_value - invested
    ret = (final_value / invested - 1.0) if invested > 0 else 0.0

    valuation_fx_rate: float | None = None
    valuation_fx_date: pd.Timestamp | None = None
    if requires_fx:
        assert usd_twd_history is not None
        valuation_fx_rate, valuation_fx_date = lookup_history_value(usd_twd_history, close.index[-1])
    else:
        valuation_fx_rate = 1.0
        valuation_fx_date = pd.Timestamp(close.index[-1])

    position_value_ntd = position_value * valuation_fx_rate
    final_value_ntd = final_value * valuation_fx_rate
    profit_ntd = final_value_ntd - invested_ntd
    ret_ntd = (final_value_ntd / invested_ntd - 1.0) if invested_ntd > 0 else 0.0

    days = max((close.index[-1] - close.index[0]).days, 1)
    years = days / 365.25
    annualized = ((final_value / invested) ** (1 / years) - 1.0) if invested > 0 and years > 0 else 0.0
    annualized_ntd = ((final_value_ntd / invested_ntd) ** (1 / years) - 1.0) if invested_ntd > 0 and years > 0 else 0.0

    return BacktestResult(
        symbol=symbol,
        invested=invested,
        position_value=position_value,
        final_value=final_value,
        profit=profit,
        return_pct=ret * 100,
        annualized_pct=annualized * 100,
        periods_executed=executed_periods,
        periods_planned=len(scheduled),
        currency="USDT" if is_crypto_symbol(symbol) else ("USD" if is_us_symbol(symbol) else "TWD"),
        invested_ntd=invested_ntd,
        position_value_ntd=position_value_ntd,
        final_value_ntd=final_value_ntd,
        profit_ntd=profit_ntd,
        return_pct_ntd=ret_ntd * 100,
        annualized_pct_ntd=annualized_ntd * 100,
        dividend_cash=dividend_cash,
        dividends_received=dividends_received,
        reinvested_units=reinvested_units,
        valuation_fx_rate=valuation_fx_rate if requires_fx else None,
        valuation_fx_date=valuation_fx_date if requires_fx else None,
        trades=trade_records,
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)
    )


def run_ema200_accumulate_backtest(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    period_days: int,
    contribution_ntd: float,
    usd_twd_rate: float,
) -> BacktestResult:
    history_start = start - pd.Timedelta(days=EMA_LOOKBACK_DAYS)
    history_close = fetch_close_series(symbol, history_start, end)
    if history_close.empty:
        raise ValueError(f"{symbol}: No trading data in the selected date range.")

    validate_history_start(symbol, start, history_close)
    close = history_close[(history_close.index >= start) & (history_close.index <= end)]
    if close.empty:
        raise ValueError(f"{symbol}: No trading data in the selected date range.")

    scheduled, aligned_schedule = build_aligned_schedule(close.index, start, end, period_days)
    if not aligned_schedule:
        raise ValueError(f"{symbol}: Unable to align any DCA dates to trading days.")

    ema200 = history_close.ewm(span=EMA_WINDOW, adjust=False, min_periods=EMA_WINDOW).mean()
    cash = 0.0
    shares = 0.0
    total_contributed = 0.0
    executed_periods = 0
    trade_records: list[TradeRecord] = []
    contribution_quote = contribution_in_quote(symbol, contribution_ntd, usd_twd_rate)
    pending_planned_dates: list[pd.Timestamp] = []

    for aligned_day in aligned_schedule:
        dt = aligned_day.trade_date
        price = float(close.loc[dt])
        contribution_total = contribution_quote * len(aligned_day.planned_dates)
        cash += contribution_total
        total_contributed += contribution_total
        pending_planned_dates.extend(aligned_day.planned_dates)

        ema_price = float(ema200.loc[dt]) if dt in ema200.index and pd.notna(ema200.loc[dt]) else np.nan
        if np.isnan(ema_price) or price > ema_price:
            continue

        units, spent, fee = calc_order_from_budget(symbol, price, cash)
        if units <= 0:
            continue

        shares += units
        cash -= spent + fee
        executed_periods += len(pending_planned_dates)
        trade_records.append(
            TradeRecord(
                planned_dates=list(pending_planned_dates),
                trade_date=dt,
                price=price,
                units=units,
                spent=spent,
                fee=fee,
            )
        )
        pending_planned_dates.clear()

    return finalize_backtest_result(
        symbol=symbol,
        strategy_label=STRATEGY_LABELS[STRATEGY_EMA200_ACCUMULATE],
        close=close,
        total_contributed=total_contributed,
        shares=shares,
        cash=cash,
        executed_periods=executed_periods,
        scheduled_count=len(scheduled),
        trade_records=trade_records,
    )


def run_dca_backtest(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    period_days: int,
    contribution_ntd: float,
    usd_twd_rate: float,
    strategy: str,
) -> BacktestResult:
    if strategy == STRATEGY_PERIODIC_ALL_IN:
        return run_periodic_all_in_backtest(symbol, start, end, period_days, contribution_ntd, usd_twd_rate)
    if strategy == STRATEGY_EMA200_ACCUMULATE:
        return run_ema200_accumulate_backtest(symbol, start, end, period_days, contribution_ntd, usd_twd_rate)
    raise ValueError(f"Unsupported strategy: {strategy}")


def write_trades_markdown(
    results: list[BacktestResult],
    start: pd.Timestamp,
    end: pd.Timestamp,
    period_days: int,
    contribution_ntd: float,
    dividend_mode: str,
) -> Path:
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.cwd() / "trade_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"dca_trades_{ts}.md"
    lines: list[str] = []
    lines.append("# DCA Trade Details")
    lines.append("")
    lines.append(f"- Date range: {start.date()} to {end.date()}")
    lines.append(f"- Strategy: {results[0].strategy_label}")
    lines.append(f"- DCA period: every {period_days} day(s)")
    lines.append(f"- Contribution per period (NTD): {contribution_ntd:,.0f}")
    lines.append(f"- Trading fee rate: {TRADING_FEE_RATE * 100:.2f}%")
    lines.append(f"- Dividend mode: {DIVIDEND_MODE_LABELS[dividend_mode]}")
    if any(r.valuation_fx_rate is not None for r in results):
        lines.append("- FX conversion: historical USD/TWD by trade date; final valuation uses the nearest available FX date on or before the asset valuation date.")
    else:
        lines.append("- FX conversion: N/A")
    lines.append("")

    for r in results:
        lines.append(f"## {r.symbol} ({r.currency})")
        lines.append(
            f"- Periods: executed {r.periods_executed} / planned {r.periods_planned}"
        )
        lines.append(f"- Total invested: {r.invested:,.0f} {r.currency}")
        lines.append(f"- Position value: {r.position_value:,.0f} {r.currency}")
        lines.append(f"- Dividends received: {r.dividends_received:,.0f} {r.currency}")
        lines.append(f"- Dividend cash on hand: {r.dividend_cash:,.0f} {r.currency}")
        lines.append(f"- Reinvested units: {format_reinvested_units(r.symbol, r.reinvested_units)}")
        lines.append(f"- Final value: {r.final_value:,.0f} {r.currency}")
        lines.append(f"- Cash remaining: {r.cash_remaining:,.2f} {r.currency}")
        lines.append(f"- Profit: {r.profit:,.0f} {r.currency}")
        if r.valuation_fx_rate is not None and r.valuation_fx_date is not None:
            lines.append(f"- Valuation USD/TWD rate: {r.valuation_fx_rate:.4f} ({r.valuation_fx_date.date()})")
            lines.append(f"- Final value (NTD): {r.final_value_ntd:,.0f} NTD")
            lines.append(f"- Profit (NTD): {r.profit_ntd:,.0f} NTD")
        lines.append("")
        lines.append("| # | Planned Date(s) | Trade Date | Price | Units | Spent | Fee |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for i, t in enumerate(r.trades, start=1):
            lines.append(
                f"| {i} | {format_planned_dates(t.planned_dates)} | {t.trade_date.date()} | "
                f"{t.price:,.2f} | {format_units(r.symbol, t.units)} | {t.spent:,.2f} | {t.fee:,.2f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DCA Backtester")
        self.resize(1120, 700)

        root = QWidget()
        root.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCentralWidget(root)
        self.root = root
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(10)

        control_box = QGroupBox("Backtest Settings")
        control_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        control_layout = QGridLayout(control_box)
        control_layout.setContentsMargins(12, 10, 12, 12)
        control_layout.setHorizontalSpacing(14)
        control_layout.setVerticalSpacing(10)

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(date(2018, 1, 1))
        self.start_date.setMinimumWidth(170)
        self.start_date.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(date.today())
        self.end_date.setMinimumWidth(170)
        self.end_date.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.period_spin = QSpinBox()
        self.period_spin.setRange(1, 3650)
        self.period_spin.setValue(30)
        self.period_spin.setMinimumWidth(180)

        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(100, 10_000_000)
        self.amount_spin.setDecimals(0)
        self.amount_spin.setValue(10_000)
        self.amount_spin.setSingleStep(1000)
        self.amount_spin.setMinimumWidth(200)

        self.dividend_mode = QComboBox()
        self.dividend_mode.addItem("Cash Dividends", DIVIDEND_MODE_CASH)
        self.dividend_mode.addItem("Auto Reinvest Dividends", DIVIDEND_MODE_REINVEST)
        self.dividend_mode.setMinimumWidth(240)

        self.display_mode = QComboBox()
        self.display_mode.addItem("Original Currency", DISPLAY_MODE_ORIGINAL)
        self.display_mode.addItem("Convert All To NTD", DISPLAY_MODE_NTD)
        self.display_mode.currentIndexChanged.connect(self.refresh_result_table)
        self.display_mode.setMinimumWidth(240)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem(STRATEGY_LABELS[STRATEGY_PERIODIC_ALL_IN], STRATEGY_PERIODIC_ALL_IN)
        self.strategy_combo.addItem(STRATEGY_LABELS[STRATEGY_EMA200_ACCUMULATE], STRATEGY_EMA200_ACCUMULATE)

        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.clicked.connect(self.on_run)
        self.run_btn.setMinimumHeight(36)
        self.run_btn.setMinimumWidth(180)
        self.clear_btn = QPushButton("Clear Output")
        self.clear_btn.clicked.connect(self.on_clear_output)
        self.clear_btn.setMinimumHeight(36)
        self.clear_btn.setMinimumWidth(180)

<<<<<<< HEAD
        control_layout.addWidget(QLabel("Start Date"), 0, 0)
        control_layout.addWidget(self.start_date, 0, 1)
        control_layout.addWidget(QLabel("End Date"), 0, 2)
        control_layout.addWidget(self.end_date, 0, 3)
        control_layout.addWidget(QLabel("DCA Period (Days)"), 1, 0)
        control_layout.addWidget(self.period_spin, 1, 1)
        control_layout.addWidget(QLabel("Amount (NTD)"), 1, 2)
        control_layout.addWidget(self.amount_spin, 1, 3)
        control_layout.addWidget(QLabel("Strategy"), 2, 0)
        control_layout.addWidget(self.strategy_combo, 2, 1, 1, 3)
        control_layout.addWidget(self.run_btn, 0, 4, 1, 1)
        control_layout.addWidget(self.clear_btn, 1, 4, 1, 1)
=======
        start_field = self.create_stacked_field("Start Date", self.start_date, align_control_left=True)
        end_field = self.create_stacked_field("End Date", self.end_date, align_control_left=True)
        period_field = self.create_stacked_field("DCA Period", self.period_spin)
        amount_field = self.create_stacked_field("Amount (NTD)", self.amount_spin)
        dividend_field = self.create_stacked_field("Dividend Mode", self.dividend_mode)
        display_field = self.create_stacked_field("Display Currency", self.display_mode)

        button_panel = QWidget()
        button_layout = QVBoxLayout(button_panel)
        button_layout.setContentsMargins(0, 22, 0, 0)
        button_layout.setSpacing(10)
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.clear_btn)
        control_layout.addWidget(start_field, 0, 0)
        control_layout.addWidget(period_field, 0, 1)
        control_layout.addWidget(amount_field, 0, 2)
        control_layout.addWidget(end_field, 1, 0)
        control_layout.addWidget(dividend_field, 1, 1)
        control_layout.addWidget(display_field, 1, 2)
        control_layout.addWidget(button_panel, 0, 3, 2, 1, Qt.AlignmentFlag.AlignTop)
        control_layout.setColumnStretch(0, 0)
        control_layout.setColumnStretch(1, 1)
        control_layout.setColumnStretch(2, 1)
        control_layout.setColumnMinimumWidth(1, 260)
        control_layout.setColumnMinimumWidth(2, 260)
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)

        layout.addWidget(control_box)

        picks_layout = QHBoxLayout()
        self.symbol_list = QListWidget()
        self.symbol_list.setSelectionMode(QListWidget.MultiSelection)
        self.symbol_list.setMaximumWidth(250)

        for group, symbols in SYMBOL_GROUPS.items():
            header = QListWidgetItem(f"[{group}]")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            self.symbol_list.addItem(header)
            for sym in symbols:
                item = QListWidgetItem(sym)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.symbol_list.addItem(item)

        picks_layout.addWidget(self.symbol_list, 1)

        right_layout = QVBoxLayout()
        self.result_table = CopyableTableWidget(0, 8)
        self.result_table.setHorizontalHeaderLabels(
            ["Symbol", "Total Invested", "Final Value", "Cash Left", "Profit", "Portfolio Return", "Annualized Return", "Periods"]
        )
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.result_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self.result_table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(False)
        header.sectionClicked.connect(self.on_result_header_clicked)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sorted_column: int | None = None
        self.sort_order = Qt.SortOrder.AscendingOrder
        self.results: list[BacktestResult] = []

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Run logs and errors will be shown here.")
        self.log_box.setMinimumHeight(90)
        self.log_box.setMaximumHeight(150)

        right_layout.addWidget(self.result_table, 5)
        right_layout.addWidget(self.log_box, 1)
        picks_layout.addLayout(right_layout, 7)

        layout.addLayout(picks_layout)
        QTimer.singleShot(0, self.reset_initial_ui_state)

    def reset_initial_ui_state(self) -> None:
        self.start_date.clearFocus()
        self.end_date.clearFocus()
        self.period_spin.clearFocus()
        self.amount_spin.clearFocus()
        self.dividend_mode.clearFocus()
        self.display_mode.clearFocus()
        self.run_btn.clearFocus()
        self.clear_btn.clearFocus()
        self.symbol_list.clearSelection()
        self.symbol_list.setCurrentRow(-1)
        self.result_table.clearSelection()
        self.root.setFocus()

    def create_stacked_field(self, label_text: str, control: QWidget, align_control_left: bool = False) -> QWidget:
        field = QWidget()
        field.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        field_layout = QVBoxLayout(field)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(4)

        label = QLabel(label_text)
        field_layout.addWidget(label)
        if align_control_left:
            field_layout.addWidget(control, 0, Qt.AlignmentFlag.AlignLeft)
        else:
            field_layout.addWidget(control)
        return field

    def selected_symbols(self) -> list[str]:
        symbols: list[str] = []
        for i in range(self.symbol_list.count()):
            item = self.symbol_list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable and item.checkState() == Qt.CheckState.Checked:
                symbols.append(item.text())
        return symbols

    def make_result_item(self, text: str, sort_value: object, align_right: bool = False) -> QTableWidgetItem:
        item = SortableTableWidgetItem(text, sort_value)
        if align_right:
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return item

    def current_display_mode(self) -> str:
        return str(self.display_mode.currentData())

    def result_table_headers(self) -> list[str]:
        if self.current_display_mode() == DISPLAY_MODE_NTD:
            return [
                "Symbol",
                "Total Invested (NTD)",
                "Final Value (NTD)",
                "Profit (NTD)",
                "Total Return",
                "Annualized Return",
                "Periods",
            ]
        return ["Symbol", "Total Invested", "Final Value", "Profit", "Total Return", "Annualized Return", "Periods"]

    def result_row_values(self, result: BacktestResult) -> list[tuple[str, object, bool]]:
        if self.current_display_mode() == DISPLAY_MODE_NTD:
            return [
                (f"{result.symbol} ({result.currency})", result.symbol, False),
                (f"{result.invested_ntd:,.0f}", result.invested_ntd, True),
                (f"{result.final_value_ntd:,.0f}", result.final_value_ntd, True),
                (f"{result.profit_ntd:,.0f}", result.profit_ntd, True),
                (f"{result.return_pct_ntd:,.0f}%", result.return_pct_ntd, True),
                (f"{result.annualized_pct_ntd:,.0f}%", result.annualized_pct_ntd, True),
                (f"{result.periods_executed}/{result.periods_planned}", (result.periods_executed, result.periods_planned), True),
            ]
        return [
            (f"{result.symbol} ({result.currency})", result.symbol, False),
            (f"{result.invested:,.0f}", result.invested, True),
            (f"{result.final_value:,.0f}", result.final_value, True),
            (f"{result.profit:,.0f}", result.profit, True),
            (f"{result.return_pct:,.0f}%", result.return_pct, True),
            (f"{result.annualized_pct:,.0f}%", result.annualized_pct, True),
            (f"{result.periods_executed}/{result.periods_planned}", (result.periods_executed, result.periods_planned), True),
        ]

    def refresh_result_table(self) -> None:
        self.result_table.setHorizontalHeaderLabels(self.result_table_headers())
        self.result_table.setRowCount(len(self.results))
        for row, result in enumerate(self.results):
            for col, (text, sort_value, align_right) in enumerate(self.result_row_values(result)):
                item = self.make_result_item(text, sort_value, align_right)
                self.result_table.setItem(row, col, item)
        if self.sorted_column is not None and self.results:
            self.result_table.sortItems(self.sorted_column, self.sort_order)

    def on_result_header_clicked(self, column: int) -> None:
        if self.sorted_column == column:
            order = (
                Qt.SortOrder.DescendingOrder
                if self.sort_order == Qt.SortOrder.AscendingOrder
                else Qt.SortOrder.AscendingOrder
            )
        else:
            order = Qt.SortOrder.AscendingOrder if column == 0 else Qt.SortOrder.DescendingOrder

        self.sorted_column = column
        self.sort_order = order
        header = self.result_table.horizontalHeader()
        header.setSortIndicatorShown(True)
        header.setSortIndicator(column, order)
        self.result_table.sortItems(column, order)

    def on_run(self) -> None:
        symbols = self.selected_symbols()
        if not symbols:
            QMessageBox.warning(self, "Warning", "Please select at least one symbol.")
            return

        start = pd.Timestamp(self.start_date.date().toPython())
        end = pd.Timestamp(self.end_date.date().toPython())

        if start >= end:
            QMessageBox.warning(self, "Warning", "Start date must be earlier than end date.")
            return

        period = self.period_spin.value()
        contribution = float(self.amount_spin.value())
<<<<<<< HEAD
        strategy = str(self.strategy_combo.currentData())
        strategy_label = self.strategy_combo.currentText()
        self.result_table.setRowCount(0)
=======
        dividend_mode = self.dividend_mode.currentData()
        self.results = []
        self.refresh_result_table()
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)
        self.log_box.clear()

        usd_twd_history: pd.Series | None = None
        has_foreign_quote_assets = any(is_us_symbol(sym) for sym in symbols)
        if has_foreign_quote_assets:
            try:
                usd_twd_history = fetch_usd_twd_history(start, end)
                self.log_box.append("[INFO] Using historical USD/TWD rates by trade date.")
            except Exception as exc:
                self.log_box.append(f"[ERR] FX rate: {exc}")
                return

<<<<<<< HEAD
        self.log_box.append(f"[INFO] Strategy: {strategy_label}")
        self.log_box.append(f"[INFO] Contribution per period: {contribution:,.0f} NTD")
        self.log_box.append(f"[INFO] Trading fee rate applied: {TRADING_FEE_RATE * 100:.2f}%")
        if strategy == STRATEGY_EMA200_ACCUMULATE:
            self.log_box.append(f"[INFO] EMA condition: buy when close <= EMA{EMA_WINDOW}.")
=======
        stock_symbols = [sym for sym in symbols if not is_crypto_symbol(sym)]
        if stock_symbols:
            try:
                prefetch_stock_close_histories(stock_symbols, start, end)
                self.log_box.append(f"[INFO] Prepared stock price history for {len(stock_symbols)} symbol(s).")
            except Exception as exc:
                self.log_box.append(f"[ERR] Stock history prefetch: {exc}")
                return
            dividend_prefetch_errors = prefetch_stock_dividend_histories(stock_symbols, start, end)
            if dividend_prefetch_errors:
                self.log_box.append(
                    f"[WARN] Dividend prefetch had {len(dividend_prefetch_errors)} issue(s); affected symbols will retry individually."
                )
            else:
                self.log_box.append(f"[INFO] Prepared dividend history for {len(stock_symbols)} symbol(s).")

        self.log_box.append(f"[INFO] Contribution per period: {contribution:,.0f} NTD")
        self.log_box.append(f"[INFO] Trading fee rate applied: {TRADING_FEE_RATE * 100:.2f}%")
        self.log_box.append(f"[INFO] Dividend mode: {DIVIDEND_MODE_LABELS[dividend_mode]}")
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)

        results: list[BacktestResult] = []
        for sym in symbols:
            try:
<<<<<<< HEAD
                res = run_dca_backtest(sym, start, end, period, contribution, usd_twd_rate, strategy)
=======
                res = run_dca_backtest(sym, start, end, period, contribution, usd_twd_history, dividend_mode)
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)
                results.append(res)
                self.log_box.append(
                    f"[OK] {sym} backtest completed. Dividends: {res.dividends_received:,.0f} {res.currency}"
                )
            except Exception as exc:
                self.log_box.append(f"[ERR] {sym}: {exc}")

        if not results:
            self.log_box.append("No result to display. Please check date range, network, and symbols.")
            return

        self.results = results
        self.refresh_result_table()

<<<<<<< HEAD
        for row, r in enumerate(results):
            values = [
                f"{r.symbol} ({r.currency})",
                f"{r.invested:,.0f}",
                f"{r.final_value:,.0f}",
                f"{r.cash_remaining:,.2f}",
                f"{r.profit:,.0f}",
                f"{r.return_pct:,.0f}%",
                f"{r.annualized_pct:,.0f}%",
                f"{r.periods_executed}/{r.periods_planned}",
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                if col > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.result_table.setItem(row, col, item)

        md_path = write_trades_markdown(results, start, end, period, contribution, usd_twd_rate)
=======
        md_path = write_trades_markdown(results, start, end, period, contribution, dividend_mode)
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)
        self.log_box.append(f"[INFO] Trade details exported: {md_path}")

    def on_clear_output(self) -> None:
        self.results = []
        self.refresh_result_table()
        self.log_box.clear()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
