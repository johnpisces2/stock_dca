# DCA Backtester

以 `Python` + `PySide6` 開發的桌面版 DCA（Dollar-Cost Averaging）回測工具。  
目前支援台股 ETF、美股 ETF、美股大型股與加密貨幣，可用固定週期投入金額進行多標的回測，並整合股利處理、歷史匯率換算、結果排序與 Markdown 交易明細報表輸出。

## Screenshot

![Application Screenshot](screenshot.png)

## 功能特色

- 多檔標的同時回測，可一次比較不同資產的 DCA 表現。
- 內建 `Taiwan ETFs`、`US ETFs`、`US Large Caps`、`Crypto` 分組清單。
- 股票資料使用 Yahoo Finance 的歷史 `Close` 作為成交與估值價格。
- 股票股利使用 `yfinance.Ticker(...).dividends` 取得，並依持有 units 計算。
- 支援 `Cash Dividends` 與 `Auto Reinvest Dividends` 兩種 Dividend Mode。
- 美股與 Crypto 依交易日歷史 `USD/TWD` 匯率換算每期投入金額。
- 可在結果表切換 `Original Currency` 或 `Convert All To NTD` 顯示。
- 結果表支援點擊欄位標題排序，數值欄位使用 numeric sorting。
- 每次成功回測後會自動輸出 Markdown 交易明細到 `trade_reports/`。
- 內建 in-memory cache，快取股票價格、股利、匯率與 Crypto OHLCV 資料，加速同 session 內重跑。

## 技術棧

- `Python`
- `PySide6`：桌面 GUI
- `pandas` / `numpy`：時間序列與回測計算
- `yfinance`：股票價格、股票股利、USD/TWD 歷史匯率
- `ccxt`：Crypto exchange OHLCV 資料

## 支援標的

目前內建標的清單定義於 `main.py` 的 `SYMBOL_GROUPS`。

### Taiwan ETFs

- `0050.TW`
- `0056.TW`
- `006208.TW`
- `00878.TW`
- `00919.TW`

### US ETFs

- `SPY`
- `QQQ`
- `TQQQ`
- `SOXX`
- `SOXL`
- `VOO`
- `VTI`

### US Large Caps

- `AAPL`
- `MSFT`
- `AMZN`
- `GOOGL`
- `META`
- `NVDA`
- `TSLA`
- `AVGO`
- `BRK.B`
- `JPM`
- `V`
- `MA`
- `LLY`
- `UNH`
- `XOM`
- `WMT`
- `COST`
- `NFLX`
- `AMD`
- `ORCL`

### Crypto

- `BTC/USDT`
- `ETH/USDT`

補充：UI 顯示 `BRK.B`，下載 Yahoo Finance 資料時會自動轉換為 `BRK-B`。

## 回測設定

GUI 提供以下輸入欄位：

- `Start Date`：回測開始日期，預設為 `2018-01-01`。
- `End Date`：回測結束日期，預設為今天。
- `DCA Period`：每隔幾天投入一次，範圍為 `1` 至 `3650` 天，預設 `30` 天。
- `Amount (NTD)`：每期投入金額，範圍為 `100` 至 `10,000,000` NTD，預設 `10,000` NTD。
- `Dividend Mode`：
  - `Cash Dividends`
  - `Auto Reinvest Dividends`
- `Display Currency`：
  - `Original Currency`
  - `Convert All To NTD`

操作按鈕：

- `Run Backtest`：執行回測。
- `Clear Output`：清空結果表與 log。

## 回測邏輯

### DCA Schedule

- 根據 `Start Date`、`End Date` 與 `DCA Period` 建立固定週期投入日期。
- 若預定投入日不是交易日，會對齊到下一個可用交易日。
- `Periods` 會以 `executed/planned` 顯示實際成交期數與原本規劃期數。

### 成交價格

- 股票使用歷史 `Close`。
- Crypto 使用 `ccxt` 取得日線 `OHLCV`，並使用 `close` 作為成交與估值價格。

### 買入單位

- 股票最小買入單位為 `0.1 share`。
- Crypto 支援 fractional units，顯示到小數 8 位。
- 若每期投入金額不足以買進股票最小單位，該期不會成交。

### Trading Fee

- 每次買入固定收取 `0.1%` trading fee。
- Trading fee 會計入 invested cost。
- 股利自動再投入目前不另外收取 trading fee。

### Dividend Mode

`Cash Dividends`

- 股利依當時持有 units 計算。
- 股利以 cash 累積。
- `Final Value = Position Value + Dividend Cash`

`Auto Reinvest Dividends`

- 股利事件會對齊到下一個可交易日。
- 若當時已有持倉，會依該交易日 `Close` 自動轉成額外 units。
- 自動再投入取得的 units 會納入後續估值。

### FX Conversion

- 台股標的以 `TWD` 計算，不需要匯率。
- 美股與 Crypto 視為外幣報價資產，會使用歷史 `USD/TWD` 匯率。
- 每期投入時，會用交易日當天或之前最近可用的 `USD/TWD` 將 NTD 轉成外幣。
- 最終 NTD 估值會使用資產估值日當天或之前最近可用的 `USD/TWD`。
- 若歷史匯率資料不足，程式不會使用未來匯率回補，以避免 look-ahead bias。

## 結果表

結果表欄位：

- `Symbol`
- `Total Invested`
- `Final Value`
- `Profit`
- `Total Return`
- `Annualized Return`
- `Periods`

顯示模式：

- `Original Currency`：依資產原始幣別顯示，台股為 `TWD`、美股為 `USD`、Crypto 為 `USDT`。
- `Convert All To NTD`：將 invested、final value、profit 與 return 轉成 NTD 視角。

排序：

- 點擊欄位標題可排序。
- Symbol 預設由小到大，其餘數值欄位預設由大到小。
- 數值欄位使用實際數值排序，不使用字串排序。

## Markdown 報表

每次成功回測後，程式會自動建立 Markdown 報表：

```text
trade_reports/dca_trades_YYYYMMDD_HHMMSS_microseconds.md
```

報表內容包含：

- 回測日期區間
- DCA period
- 每期投入金額
- Trading fee rate
- Dividend mode
- FX conversion 說明
- 每檔標的 summary
- 股利、再投入 units、最終估值與損益
- 每筆交易明細：`planned date`、`trade date`、`price`、`units`、`spent`、`fee`

若本次回測沒有使用外幣換算，報表會顯示 `FX conversion: N/A`。

## 資料來源

- 股票價格：`yfinance.download(...)`
- 股票股利：`yfinance.Ticker(...).dividends`
- USD/TWD 匯率：Yahoo Finance symbol `TWD=X`
- Crypto OHLCV：`ccxt`

Crypto 資料會依序嘗試以下 exchange：

- `binance`
- `kraken`
- `coinbase`

## 快取與效能

目前程式在單次 app session 內使用 in-memory cache：

- `STOCK_CLOSE_CACHE`
- `STOCK_DIVIDEND_CACHE`
- `CRYPTO_HISTORY_CACHE`
- `FX_HISTORY_CACHE`
- `FIRST_AVAILABLE_DATE_CACHE`

因此第一次 cold run 需要下載資料；相同條件下重跑通常會明顯加快。關閉 app 後 cache 會消失。

## 安裝

建議使用 virtual environment：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 執行

```bash
python main.py
```

啟動後在左側勾選標的，設定回測日期、DCA period、每期投入金額與 Dividend Mode，按下 `Run Backtest` 即可執行。

## 專案結構

```text
.
├── main.py
├── requirements.txt
├── screenshot.png
├── trade_reports/
└── README.md
```

主要檔案：

- `main.py`：GUI、資料下載、cache、回測邏輯、股利處理、匯率處理與 Markdown 報表輸出。
- `requirements.txt`：Python package dependencies。
- `trade_reports/`：回測產出的 Markdown 交易明細。

## 注意事項

- 本工具僅供歷史資料回測與研究使用，不構成投資建議。
- 回測結果受資料來源品質、缺值、匯率資料可用日期與交易日對齊方式影響。
- Yahoo Finance 與各 Crypto exchange API 可能有 rate limit 或暫時性資料錯誤。
- `Auto Reinvest Dividends` 使用股利事件對齊後的交易日 close 進行再投入，與實際券商入帳時間可能不同。
