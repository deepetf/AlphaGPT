# AlphaGPT - Agent Development Guide

## 语言偏好
永远使用中文交互和写文档

## Project Overview
AlphaGPT is a symbolic regression framework for quantitative trading, using Deep Learning to discover high-order, non-linear alpha factors. The system targets crypto markets (Solana) with capabilities for data ingestion, backtesting, and automated execution.

## Core Architecture
```
model_core/        - Deep learning models & symbolic regression engine
data_pipeline/     - Async data fetching & PostgreSQL storage
execution/         - Solana trading execution (Jupiter integration)
strategy_manager/  - Portfolio management & risk controls
dashboard/         - Visualization & monitoring
```

## Build Commands
```bash
# No formal build step required (Python)
# Run main training:
python -m model_core.engine

# Run data pipeline:
python -m data_pipeline.run_pipeline

# Start strategy runner:
python -m strategy_manager.runner
```

## Testing
**No test suite currently exists.** When adding tests, use pytest:
```bash
# Run all tests
pytest

# Run single test file
pytest tests/test_vm.py

# Run specific test
pytest tests/test_vm.py::test_execute_formula

# With coverage
pytest --cov=model_core
```

## Code Style Guidelines

### Imports
Order: stdlib → third-party → local (relative imports for intra-package)
```python
import torch
from loguru import logger
import asyncpg

from .config import ModelConfig
from .ops import OPS_CONFIG
```

### Naming Conventions
- **Classes**: PascalCase (AlphaGPT, PortfolioManager, RiskEngine)
- **Functions/Methods**: snake_case (get_token_history, calculate_position_size)
- **Constants**: UPPER_SNAKE_CASE (MAX_FORMULA_LEN, DB_USER)
- **Private members**: single underscore prefix (_PRIV_KEY_STR)

### Type Hints
Use type hints sparingly on public interfaces:
```python
async def buy(self, token_address: str, amount_sol: float, slippage_bps=500):
    ...
```

### Configuration
Use dataclasses for config, load from environment variables with defaults:
```python
@dataclass
class Position:
    token_address: str
    entry_price: float

class Config:
    DB_USER = os.getenv("DB_USER", "postgres")
```

### Error Handling
- Use try/except for external API calls, database operations
- Return None for recoverable failures (e.g., fetcher.py)
- Use loguru for structured logging:
  - `logger.info()` - Normal operations
  - `logger.warning()` - Recoverable issues
  - `logger.error()` - Failures
  - `logger.success()` - Successful outcomes

### Async/Await
All IO operations (database, API calls, RPC) use asyncio:
```python
async def fetch_data(self, session, address):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()
```

### Data Structures
- PyTorch tensors for numerical data
- Dataclasses for structured records
- Dictionaries for simple key-value storage
- JSON for persistence (portfolio state)

### Database
Use asyncpg with connection pooling:
```python
async with self.pool.acquire() as conn:
    await conn.execute("INSERT INTO ...")
```

### ML/PyTorch
- Use DataLoader patterns for data
- Move tensors to proper device (CPU/GPU)
- Handle NaN/Inf in intermediate computations
- Use torch.nn for model definitions

## File Organization
Each module has:
- `config.py` - Configuration class
- `__init__.py` - Empty or exports
- Main logic classes in separate files
- Entry points with `if __name__ == "__main__":`

## Key Dependencies
- torch - Deep learning
- asyncpg - PostgreSQL async driver
- aiohttp - Async HTTP client
- loguru - Logging
- solders/solana-py - Solana SDK
- dotenv - Environment variables

## Common Patterns
1. **Context managers** for database connections
2. **Semaphore** for rate limiting async requests
3. **Progress bars** using tqdm for long operations
4. **State persistence** via JSON for portfolio/trading state
5. **Causal masking** for transformer models (autoregressive)
