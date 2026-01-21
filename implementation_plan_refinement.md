Parametric Backtest & Strategy Optimization Update
====================================================

This update replaces the static backtesting engine with a dynamic one that optimizes both strategy logic and key parameters (RSI Period, ATR Multiplier).

Changes
-------

### 1. New `backtest_engine`
- **Dynamic Parameters**: Now accepts a `params` dictionary to configure `rsi_period` and `sl_mult`.
- **Logic**:
  - Calculates indicators using the provided `rsi_period`.
  - Simulates 'TREND', 'REVERSION', and 'BREAKOUT' strategies.
  - Supports dynamic trailing stops based on `sl_mult`.
- **Output**: Returns performance metrics including `pnl`, `win_rate`, and the utilized `params`.

### 2. Enhanced `find_best_strategy`
- **Grid Search**: Iterates through a refined parameter grid:
  - Standard: RSI 14, SL 2.5
  - Heavy/Slow Stocks: RSI 21, SL 3.0
  - Fast Stocks: RSI 9, SL 2.0
- **Scoring**:
  - Prioritizes stability: `PnL + (WinRate * 0.4)`.
  - Looser penalty for low trade counts (`trades < 3`) to avoid filtering out valid but infrequent setups.
- **Profitability Check**: Ensures the selected strategy is actually profitable (PnL > -5% tolerance).

### 3. Integration with UI & Scanning
- **`scan_single_stock`**:
  - Uses the optimized `rsi_period` when fetching advanced data.
  - Uses the optimized `sl_mult` (as `atr_mult`) for scoring and risk levels.
- **Analysis Tab**:
  - Displays the identified best strategy.
  - recalculates the smart score using the specific optimized parameters for that stock.

Files Modified
--------------
- `app.py`
