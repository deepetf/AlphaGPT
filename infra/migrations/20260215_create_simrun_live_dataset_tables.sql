-- sim_run live dataset SQL tables
-- Target DB: CB_HISTORY (MySQL 8+)

CREATE TABLE IF NOT EXISTS sim_live_nav_history (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  strategy_id VARCHAR(64) NOT NULL COMMENT 'strategy id',
  trade_date DATE NOT NULL COMMENT 'trading date',
  nav DECIMAL(20,4) NOT NULL COMMENT 'cash + holdings value',
  cash DECIMAL(20,4) NOT NULL COMMENT 'cash balance',
  holdings_value DECIMAL(20,4) NOT NULL COMMENT 'portfolio market value',
  holdings_count INT NOT NULL COMMENT 'position count',
  daily_ret DECIMAL(18,8) NOT NULL COMMENT 'daily return',
  cum_ret DECIMAL(18,8) NOT NULL COMMENT 'cumulative return',
  mdd DECIMAL(18,8) NOT NULL COMMENT 'max drawdown snapshot',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_live_nav_strategy_date (strategy_id, trade_date),
  KEY idx_live_nav_trade_date (trade_date),
  KEY idx_live_nav_strategy (strategy_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS sim_live_daily_holdings (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  strategy_id VARCHAR(64) NOT NULL COMMENT 'strategy id',
  trade_date DATE NOT NULL COMMENT 'trading date',
  code VARCHAR(32) NOT NULL COMMENT 'convertible bond code',
  name VARCHAR(128) NOT NULL COMMENT 'security name',
  shares BIGINT NOT NULL COMMENT 'holding shares/lot count',
  avg_cost DECIMAL(20,6) NOT NULL COMMENT 'average entry cost',
  last_price DECIMAL(20,6) NOT NULL COMMENT 'valuation price',
  entry_date DATE NULL COMMENT 'first entry date',
  market_value DECIMAL(20,4) NOT NULL COMMENT 'shares * last_price',
  pnl DECIMAL(20,4) NOT NULL COMMENT 'floating pnl',
  pnl_pct DECIMAL(18,8) NOT NULL COMMENT 'floating pnl pct',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_live_holding_strategy_date_code (strategy_id, trade_date, code),
  KEY idx_live_holding_trade_date (trade_date),
  KEY idx_live_holding_code_date (code, trade_date),
  KEY idx_live_holding_strategy (strategy_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS sim_live_trade_history (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  strategy_id VARCHAR(64) NOT NULL COMMENT 'strategy id',
  trade_date DATE NOT NULL COMMENT 'trading date',
  trade_time DATETIME NOT NULL COMMENT 'execution time',
  code VARCHAR(32) NOT NULL COMMENT 'convertible bond code',
  name VARCHAR(128) NOT NULL COMMENT 'security name',
  side VARCHAR(16) NOT NULL COMMENT 'BUY / SELL / SELL-TP',
  shares BIGINT NOT NULL COMMENT 'executed shares/lot count',
  price DECIMAL(20,6) NOT NULL COMMENT 'execution price',
  amount DECIMAL(20,4) NOT NULL COMMENT 'BUY total cost, SELL net proceeds',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_live_trade_strategy_date (strategy_id, trade_date),
  KEY idx_live_trade_code_date (code, trade_date),
  KEY idx_live_trade_side_date (side, trade_date),
  KEY idx_live_trade_time (trade_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
