
import multiprocessing
import os
import sys

# Ensure we can import model_core
sys.path.append(os.getcwd())

def worker_func():
    try:
        from model_core.config import RobustConfig
        print(f"[Worker] Loading Config...")
        print(f"[Worker] JACCARD_THRESHOLD: {RobustConfig.JACCARD_THRESHOLD}")
        print(f"[Worker] DENSITY_PENALTY: {RobustConfig.DENSITY_PENALTY}")
        print(f"[Worker] MAX_TS_IN_WINDOW: {RobustConfig.MAX_TS_IN_WINDOW}")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("[Main] Starting worker test...")
    # Use spawn to mimic Windows behavior (or fork if on Linux, but safe to force spawn for test)
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=worker_func)
    p.start()
    p.join()
    print("[Main] Worker test finished.")
