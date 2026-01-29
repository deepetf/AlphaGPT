
import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("debug_sim.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("SimulationTest")

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.append(project_root)

from strategy_manager.cb_runner import CBStrategyRunner
from strategy_manager.cb_portfolio import CBPortfolioManager

def run_simulation():
    logger.info("="*50)
    logger.info("   CB STRATEGY RUNNER: SYSTEM SIMULATION TEST")
    logger.info("="*50)

    # 1. Check Initial Portfolio State
    portfolio = CBPortfolioManager()
    logger.info("[Pre-Run] Checking Portfolio State...")
    initial_codes = portfolio.get_position_codes()
    initial_value = portfolio.get_holdings_value()
    logger.info(f"Initial Positions: {len(initial_codes)} - Value: {initial_value:.2f}")

    # 2. Run Strategy with Simulation Mode Enabled
    logger.info("\n[Run] Executing Strategy with simulate=True...")
    runner = CBStrategyRunner()
    
    # Load default strategy
    if not runner.load_strategy():
        logger.error("Failed to load strategy properly. Aborting.")
        return

    # Execute
    try:
        runner.run(simulate=True)
    except Exception as e:
        logger.error(f"Simulation run failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Check Post-Run Portfolio State
    logger.info("\n[Post-Run] Checking Portfolio State...")
    # Reload portfolio state from disk
    portfolio.load_state() 
    final_codes = portfolio.get_position_codes()
    final_value = portfolio.get_holdings_value()
    
    logger.info(f"Final Positions: {len(final_codes)} - Value: {final_value:.2f}")
    
    if len(final_codes) > 0 and len(final_codes) != len(initial_codes):
        logger.info("✅ SUCCESS: Portfolio positions updated (Closed Loop verified).")
    elif len(final_codes) == len(initial_codes):
         logger.info("⚠️ NOTICE: Portfolio positions count unchanged. (Maybe no orders generated or existing holdings matched target)")
    else:
        logger.info("❓ UNKNOWN: Please check logs for details.")

    logger.info(f"Simulation Test Completed at {datetime.now()}")

if __name__ == "__main__":
    run_simulation()
