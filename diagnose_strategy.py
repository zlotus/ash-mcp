import asyncio
import baostock as bs
from loguru import logger
from tools.strategy import get_value_candidates_and_grid_impl
from models.inputs import ValueCandidatesGridInput

async def diagnose_logic():
    # Use standard parameters to verify the actual tool implementation
    params = ValueCandidatesGridInput(
        anchor_symbol="601318", # 中国平安
        top_n=3,
        candidate_limit=60
    )
    
    logger.info(f"--- Diagnosing actual implementation for {params.anchor_symbol} ---")
    try:
        result = await get_value_candidates_and_grid_impl(params)
        logger.success("Diagnostic completed.")
        print("\n--- Diagnostic Result ---")
        print(result)
    except Exception as e:
        logger.exception(f"Diagnostic failed: {e}")
    finally:
        try:
            bs.logout()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(diagnose_logic())