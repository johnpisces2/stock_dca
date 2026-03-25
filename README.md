# DCA Backtester

以 Python + PySide6 製作的桌面版 DCA (Dollar-Cost Averaging) 回測工具，支援台股 ETF、美股 ETF、美股大型股與加密貨幣，並提供股利模式、歷史匯率換算、結果排序與 Markdown 報表輸出。

## Screenshot

請將最新介面截圖放在專案根目錄的 `screenshot.png`，README 會顯示這張圖：

![Application Screenshot](screenshot.png)

## 功能特色

- 支援多檔標的同時回測。
- 支援 `Taiwan ETFs`、`US ETFs`、`US Large Caps`、`Crypto` 分組。
- 股票使用原始 `Close` 作為成交價。
- 股票股利使用 `Ticker.dividends` 個別計算，不再依賴 `Adj Close` 近似。
- 支援兩種 `Dividend Mode`：
  - `Cash Dividends`
  - `Auto Reinvest Dividends`
- 美股與加密資產使用歷史 `USD/TWD` 依交易日換算，不使用單一最新匯率。
- 結果表支援 `Original Currency` / `Convert All To NTD` 顯示模式切換。
- 結果表支援點擊欄位標題排序。
- 每次回測後自動輸出 Markdown 交易報表。
- 內建 in-memory cache，會快取股價、股利、匯率與 crypto 歷史資料，加速重跑。

## 支援標的

目前內建清單如下：

- `Taiwan ETFs`
  - `0050.TW`
  - `0056.TW`
  - `006208.TW`
  - `00878.TW`
  - `00919.TW`
- `US ETFs`
  - `SPY`
  - `QQQ`
  - `TQQQ`
  - `SOXX`
  - `SOXL`
  - `VOO`
  - `VTI`
- `US Large Caps`
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
- `Crypto`
  - `BTC/USDT`
  - `ETH/USDT`

補充：

- UI 顯示 `BRK.B`，下載 Yahoo Finance 資料時會自動轉成 `BRK-B`。

## 回測設定

介面提供以下輸入欄位：

- `Start Date`
- `End Date`
<<<<<<< HEAD
- `DCA Period (Days)` (every N day(s), default: 30)
- `Amount (NTD)` (integer display)
- `Strategy`

Strategies:
- `Periodic Buy (Use available cash every cycle)`: add fixed capital every cycle and buy immediately with current available cash.
- `EMA200 Trigger (Accumulate cash until price <= EMA200)`: add fixed capital every cycle, but only deploy accumulated cash when the trade-day close is less than or equal to the daily `EMA200`.
=======
- `DCA Period (Days)`
- `Amount (NTD)`
- `Dividend Mode`
- `Display Currency`
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)

按鈕：

<<<<<<< HEAD
1. Build a planned DCA schedule from start to end date.
2. Align each planned date to the next available trading day.
3. If multiple planned dates align to the same trading day, their capital is merged into one order and only one trading fee is charged.
4. For Taiwan symbols (`.TW`): use the NTD amount directly.
5. For US symbols: convert NTD amount to USD using current `USD/TWD` rate (`TWD=X`).
6. Each cycle adds the fixed amount into strategy cash.
7. Stocks: buy in 0.1-share increments. Crypto: buy fractional coin amounts.
8. Trading fee is fixed at `0.1%` per buy for both stocks and crypto.
9. `Periodic Buy` spends available cash on every aligned trade date, and if cash is not enough to buy the minimum size, it rolls forward to later periods.
10. `EMA200 Trigger` uses daily close data to compute `EMA200`, and only buys when `close <= EMA200`.
11. Final metrics are calculated from accumulated shares, remaining cash, and the last available adjusted close.
=======
- `Run Backtest`
- `Clear Output`
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)

## 回測邏輯

### 1. DCA schedule

- 根據 `Start Date`、`End Date`、`DCA Period (Days)` 建立定期投入日期。
- 每個預定投入日會對齊到下一個可交易日。

### 2. 成交價格

- 股票使用歷史 `Close`。
- Crypto 使用 `ccxt` 的日線 `OHLCV close`。

### 3. 股數規則

- 股票以 `0.1 share` 為最小買入單位。
- Crypto 支援 fractional units。

### 4. Trading fee

- 每次買入固定收取 `0.1%` fee。

### 5. 股利模式

`Cash Dividends`

- 股利以現金累積。
- `Final Value = Position Value + Dividend Cash`

`Auto Reinvest Dividends`

- 股利在股利事件對應的交易日，依當天 `Close` 自動轉成更多 units。
- 預設不另外收取再投入交易 fee。

### 6. 匯率邏輯

- 台股資產直接以 `NTD` 計算。
- 美股與目前的 crypto 報價資產，會以歷史 `USD/TWD` 在每次交易日換算投入金額。
- 最終 `NTD` 估值會使用資產估值日當天，若無資料則往前最近可用的 `USD/TWD`。

## 結果表

結果表欄位：

- `Symbol`
- `Total Invested`
- `Final Value`
- `Cash Left`
- `Profit`
- `Portfolio Return`
- `Annualized Return`
- `Periods`

說明：

- `Periods` 顯示格式為 `executed/planned`
- 點擊欄位標題可排序
- 數值欄位使用數值排序，不是字串排序
- `Display Currency = Original Currency` 時，顯示各資產原本幣別
- `Display Currency = Convert All To NTD` 時，金額與報酬會改用 `NTD` 視角顯示

## 報表輸出

每次成功回測後，程式會輸出 Markdown 報表到：

```text
trade_reports/
```

<<<<<<< HEAD
Report includes:
- run settings
- selected strategy
- FX rate used
- trading fee rate used
- per-symbol summary
- per-trade details (`planned date`, `trade date`, `price`, `units`, `spent`, `fee`)
=======
檔名格式：
>>>>>>> 72ce36b (Enhance DCA Backtester with Dividend Handling and Currency Conversion)

```text
dca_trades_YYYYMMDD_HHMMSS.md
```

報表內容包含：

- 回測日期區間
- `DCA period`
- 每期投入 `NTD`
- `Trading fee rate`
- `Dividend mode`
- `FX conversion` 說明
- 每檔標的的 summary
- 每筆交易的明細

若該次回測沒有用到外幣換算，報表會顯示：

```text
FX conversion: N/A
```

## 資料來源

- 股票與匯率：`yfinance`
- 股票股利：`yfinance.Ticker(...).dividends`
- Crypto：`ccxt`

Crypto 預設會優先嘗試：

- Binance
- Kraken
- Coinbase

## 效能優化

目前已實作以下優化：

- 股票價格支援 batch download
- 股票股利支援 prefetch + parallel fetch
- 歷史 `USD/TWD` 支援 cache
- 同條件重跑會直接吃 in-memory cache

這代表：

- 第一次 cold run 仍需下載資料
- 同一個 app session 內重新回測通常會快很多

## 安裝

```bash
pip install -r requirements.txt
```

## 執行

```bash
python main.py
```

## 專案檔案

- `main.py`
  - GUI
  - 回測邏輯
  - 匯率處理
  - 股利處理
  - 資料快取
- `requirements.txt`
  - 套件依賴
- `trade_reports/`
  - 匯出的 Markdown 報表
- `screenshot.png`
  - README 使用的畫面截圖

## 注意事項

- 本工具屬於 historical backtest，不代表未來績效。
- `USDT` 目前以 `USD/TWD` 作近似換算，適合一般比較用途，但不是獨立的 stablecoin FX model。
- 股利資料是否完整，取決於資料來源提供狀況。
- 報表檔名目前使用秒級時間戳，若同一秒連續匯出，可能覆蓋前一份檔案。
