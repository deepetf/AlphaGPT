# AlphaGPT Changelog

## [V4.1.2] - 2026-02-05

### Added
- **SimFailShare Metric**: Now correctly calculates the redundancy rate of *good* factors (SimReject / MetricPass), decoupling it from structural failures.
- **V4.X Architecture Specs**: Archived implementation plans for V4.0, V4.1, V4.1.1, V4.1.2 in `doc/`.

### Fixed
- **SimPass Logic**: Explicitly includes `SIM_REPLACE` (better factors replacing old ones) as a success event.
- **Code Cleanup**: Removed deprecated `beta_is_locked` variables to reduce confusion.

## [V4.1] - 2026-02-04 (Low Risk Online)

### Key Features
- **Rolling Window Controller**: Entropy beta now reacts to `RollingHardPassRate` (10-step window) instead of volatile single-step rates, preventing erratic resets.
- **3-Level Success Criteria**: Introduced `HardPass` (Valid), `MetricPass` (Profitable), and `SimPass` (Unique) to precisely diagnose pipeline bottlenecks.
- **Dynamic Penalty Lock**: If `LowVar` failure persists (>70% for 50 steps), penalty multiplier increases (x1.15) to force exploration.

### Performance
- **Zero Struct Failure**: Achieved <0.2% error rate via Grammar-Guided Masking.
- **Diversity**: Maintains ~5% healthy redundancy (`SimReject`) without collapsing into mode repetition.

## [V4.0] - 2026-02-01 (Robustness Overhaul)

### Key Features
- **Hierarchy of Failure**: Implemented tiered penalties (`EXEC` -10, `STRUCT` -8, `LOWVAR` -6.5, `METRIC` -4~-6) to create a learnable gradient for RL.
- **Safe LRU Cache**: 100k capacity `OrderedDict` to eliminate redundant backtests within and across batches.
- **Observability 2.0**: Added console logs for `Gap`, `RStd`, and `TopFail` distribution.
