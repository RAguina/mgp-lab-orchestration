from langchain_integration.wrappers.hf_pipeline_wrappers import (
    standard,
    optimized,
    streaming,
)

STRATEGY_LOADERS = {
    "standard": standard.load_model,
    "optimized": optimized.load_model,
    "streaming": streaming.load_model,
}
