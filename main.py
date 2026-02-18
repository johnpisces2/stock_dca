from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import yfinance as yf
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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


@dataclass
class BacktestResult:
    symbol: str
    invested: float
    final_value: float
    profit: float
    return_pct: float
    annualized_pct: float
    periods_executed: int
    periods_planned: int
    currency: str
    trades: list["TradeRecord"]


@dataclass
class TradeRecord:
    planned_date: pd.Timestamp
    trade_date: pd.Timestamp
    price: float
    units: float
    spent: float
    fee: float


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


def fetch_crypto_close_series(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
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
            return close
        except Exception as exc:
            errors.append(f"{exchange_id}: {exc}")

    err = "; ".join(errors) if errors else "unknown error"
    raise ValueError(f"{symbol}: Unable to fetch crypto history via ccxt ({err}).")


def fetch_stock_close_series(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    yf_symbol = to_yf_symbol(symbol)
    df = yf.download(
        yf_symbol,
        start=(start - pd.Timedelta(days=10)).date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        auto_adjust=False,
        progress=False,
        actions=False,
    )

    if df.empty:
        raise ValueError(f"{symbol}: No historical data found.")

    if isinstance(df.columns, pd.MultiIndex):
        close = (
            df[("Adj Close", yf_symbol)]
            if ("Adj Close", yf_symbol) in df.columns
            else df[("Close", yf_symbol)]
        )
    else:
        close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]

    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close[(close.index >= start) & (close.index <= end)]
    return close


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


def fetch_close_series(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if is_crypto_symbol(symbol):
        return fetch_crypto_close_series(symbol, start, end)
    return fetch_stock_close_series(symbol, start, end)


def format_units(symbol: str, units: float) -> str:
    if is_crypto_symbol(symbol):
        return f"{units:.8f}"
    return f"{units:.1f}"


def validate_history_start(symbol: str, start: pd.Timestamp, close: pd.Series, grace_days: int = 40) -> None:
    if close.empty:
        return
    first = pd.Timestamp(close.index[0])
    if first > (start + pd.Timedelta(days=grace_days)):
        raise ValueError(
            f"{symbol}: Start date is {start.date()}, but earliest available data is {first.date()}. "
            "Please move the backtest start date forward."
        )


def run_dca_backtest(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    period_days: int,
    contribution_ntd: float,
    usd_twd_rate: float,
) -> BacktestResult:
    close = fetch_close_series(symbol, start, end)

    if close.empty:
        raise ValueError(f"{symbol}: No trading data in the selected date range.")
    validate_history_start(symbol, start, close)

    scheduled = generate_schedule(start, end, period_days)
    aligned_schedule: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for planned_dt in scheduled:
        pos = close.index.searchsorted(planned_dt)
        if pos < len(close.index):
            aligned_schedule.append((planned_dt, close.index[pos]))

    if not aligned_schedule:
        raise ValueError(f"{symbol}: Unable to align any DCA dates to trading days.")

    shares = 0.0
    invested = 0.0
    executed_periods = 0
    trade_records: list[TradeRecord] = []

    for planned_dt, dt in aligned_schedule:
        price = float(close.loc[dt])
        if price <= 0:
            continue
        contribution_in_quote = contribution_ntd if not is_us_symbol(symbol) else (contribution_ntd / usd_twd_rate)
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

    if shares <= 0 or invested <= 0:
        if is_crypto_symbol(symbol):
            raise ValueError(f"{symbol}: Amount (NTD) is too low to buy crypto in selected periods.")
        raise ValueError(f"{symbol}: Amount (NTD) is too low to buy at least 0.1 share per DCA period.")

    last_price = float(close.iloc[-1])
    final_value = shares * last_price
    profit = final_value - invested
    ret = (final_value / invested - 1.0) if invested > 0 else 0.0

    days = max((close.index[-1] - close.index[0]).days, 1)
    years = days / 365.25
    annualized = ((final_value / invested) ** (1 / years) - 1.0) if invested > 0 and years > 0 else 0.0

    return BacktestResult(
        symbol=symbol,
        invested=invested,
        final_value=final_value,
        profit=profit,
        return_pct=ret * 100,
        annualized_pct=annualized * 100,
        periods_executed=executed_periods,
        periods_planned=len(scheduled),
        currency="USDT" if is_crypto_symbol(symbol) else ("USD" if is_us_symbol(symbol) else "TWD"),
        trades=trade_records,
    )


def write_trades_markdown(
    results: list[BacktestResult],
    start: pd.Timestamp,
    end: pd.Timestamp,
    period_days: int,
    contribution_ntd: float,
    usd_twd_rate: float,
) -> Path:
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.cwd() / "trade_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"dca_trades_{ts}.md"
    lines: list[str] = []
    lines.append("# DCA Trade Details")
    lines.append("")
    lines.append(f"- Date range: {start.date()} to {end.date()}")
    lines.append(f"- DCA period: every {period_days} day(s)")
    lines.append(f"- Contribution per period (NTD): {contribution_ntd:,.0f}")
    lines.append(f"- USD/TWD rate used: {usd_twd_rate:.4f}")
    lines.append(f"- Trading fee rate: {TRADING_FEE_RATE * 100:.2f}%")
    lines.append("")

    for r in results:
        lines.append(f"## {r.symbol} ({r.currency})")
        lines.append(
            f"- Periods: executed {r.periods_executed} / planned {r.periods_planned}"
        )
        lines.append(f"- Total invested: {r.invested:,.0f} {r.currency}")
        lines.append(f"- Final value: {r.final_value:,.0f} {r.currency}")
        lines.append(f"- Profit: {r.profit:,.0f} {r.currency}")
        lines.append("")
        lines.append("| # | Planned Date | Trade Date | Price | Units | Spent | Fee |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for i, t in enumerate(r.trades, start=1):
            lines.append(
                f"| {i} | {t.planned_date.date()} | {t.trade_date.date()} | "
                f"{t.price:,.2f} | {format_units(r.symbol, t.units)} | {t.spent:,.2f} | {t.fee:,.2f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DCA Backtester (Dividend-Adjusted)")
        self.resize(1120, 700)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        control_box = QGroupBox("Backtest Settings")
        control_layout = QGridLayout(control_box)

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(date(2018, 1, 1))

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(date.today())

        self.period_spin = QSpinBox()
        self.period_spin.setRange(1, 3650)
        self.period_spin.setValue(30)

        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(100, 10_000_000)
        self.amount_spin.setDecimals(0)
        self.amount_spin.setValue(10_000)
        self.amount_spin.setSingleStep(1000)

        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.clicked.connect(self.on_run)
        self.clear_btn = QPushButton("Clear Output")
        self.clear_btn.clicked.connect(self.on_clear_output)

        control_layout.addWidget(QLabel("Start Date"), 0, 0)
        control_layout.addWidget(self.start_date, 0, 1)
        control_layout.addWidget(QLabel("End Date"), 0, 2)
        control_layout.addWidget(self.end_date, 0, 3)
        control_layout.addWidget(QLabel("DCA Period (Days)"), 1, 0)
        control_layout.addWidget(self.period_spin, 1, 1)
        control_layout.addWidget(QLabel("Amount (NTD)"), 1, 2)
        control_layout.addWidget(self.amount_spin, 1, 3)
        control_layout.addWidget(self.run_btn, 0, 4, 1, 1)
        control_layout.addWidget(self.clear_btn, 1, 4, 1, 1)

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
        self.result_table = QTableWidget(0, 7)
        self.result_table.setHorizontalHeaderLabels(
            ["Symbol", "Total Invested", "Final Value", "Profit", "Total Return", "Annualized Return", "Periods"]
        )
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.result_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Run logs and errors will be shown here.")

        right_layout.addWidget(self.result_table, 3)
        right_layout.addWidget(self.log_box, 2)
        picks_layout.addLayout(right_layout, 7)

        layout.addLayout(picks_layout)

    def selected_symbols(self) -> list[str]:
        symbols: list[str] = []
        for i in range(self.symbol_list.count()):
            item = self.symbol_list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable and item.checkState() == Qt.CheckState.Checked:
                symbols.append(item.text())
        return symbols

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
        self.result_table.setRowCount(0)
        self.log_box.clear()

        usd_twd_rate = 1.0
        has_usd_quote_assets = any(is_us_symbol(sym) for sym in symbols)
        if has_usd_quote_assets:
            try:
                usd_twd_rate = fetch_usd_twd_rate()
                self.log_box.append(f"[INFO] Using USD/TWD rate: {usd_twd_rate:.4f}")
            except Exception as exc:
                self.log_box.append(f"[ERR] FX rate: {exc}")
                return

        self.log_box.append(f"[INFO] Contribution per period: {contribution:,.0f} NTD")
        self.log_box.append(f"[INFO] Trading fee rate applied: {TRADING_FEE_RATE * 100:.2f}%")

        results: list[BacktestResult] = []
        for sym in symbols:
            try:
                res = run_dca_backtest(sym, start, end, period, contribution, usd_twd_rate)
                results.append(res)
                self.log_box.append(f"[OK] {sym} backtest completed.")
            except Exception as exc:
                self.log_box.append(f"[ERR] {sym}: {exc}")

        if not results:
            self.log_box.append("No result to display. Please check date range, network, and symbols.")
            return

        self.result_table.setRowCount(len(results))

        for row, r in enumerate(results):
            values = [
                f"{r.symbol} ({r.currency})",
                f"{r.invested:,.0f}",
                f"{r.final_value:,.0f}",
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
        self.log_box.append(f"[INFO] Trade details exported: {md_path}")

    def on_clear_output(self) -> None:
        self.result_table.setRowCount(0)
        self.log_box.clear()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
