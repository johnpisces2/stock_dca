# DCA Backtester (PySide6)

A desktop DCA (Dollar-Cost Averaging) backtesting app built with Python and PySide6.

![Application Screenshot](screenshot.png)

## What It Does

- Backtests periodic investing with dividend-adjusted historical prices (`Adj Close` from Yahoo Finance).
- Supports integer-share buying only (no fractional shares).
- Lets you backtest multiple symbols in one run.
- Exports detailed trade logs as Markdown after each run.

## Symbol Groups

- Taiwan ETFs
- US ETFs
- US Large Caps

Notes:
- `BRK.B` is shown in the UI, but automatically mapped to Yahoo Finance ticker `BRK-B` when downloading data.

## Inputs

- `Start Date`
- `End Date`
- `DCA Period` (every N month(s), range: 1 to 24)
- `Amount (NTD)` (integer display)

## Backtest Logic

1. Build a planned DCA schedule from start to end date.
2. Align each planned date to the next available trading day.
3. For Taiwan symbols (`.TW`): use the NTD amount directly.
4. For US symbols: convert NTD amount to USD using current `USD/TWD` rate (`TWD=X`).
5. Buy integer shares only for each period.
6. If a period cannot buy at least 1 share, that period is skipped.
7. Final metrics are calculated from accumulated shares and the last available adjusted close.

## Result Table

Columns:
- `Symbol` (includes currency tag, e.g. `SPY (USD)`, `0050.TW (TWD)`)
- `Total Invested`
- `Final Value`
- `Profit`
- `Total Return`
- `Annualized Return`
- `Periods` (`executed/planned`)

## Trade Report Output

After each successful run, the app writes a Markdown report to:

- `trade_reports/`

Filename format:

- `dca_trades_YYYYMMDD_HHMMSS.md`

Report includes:
- run settings
- FX rate used
- per-symbol summary
- per-trade details (`planned date`, `trade date`, `price`, `units`, `spent`)

## UI Actions

- `Run Backtest`: runs backtest for all selected symbols.
- `Clear Output`: clears both the result table and log panel.

## Requirements

- Python 3.10+
- Internet connection (Yahoo Finance data + FX rate)

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Project Files

- `main.py`: GUI + backtest engine
- `requirements.txt`: dependencies
- `.gitignore`: excludes `__pycache__/` and `trade_reports/`
