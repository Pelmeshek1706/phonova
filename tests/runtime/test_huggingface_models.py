import os

import pytest

from tests.helpers.runtime_checks import (
    check_embedding_gemma_model,
    check_gemma_causal_lm,
    check_hf_sentiment_pipeline,
)


def test_xlm_roberta_sentiment_pipeline_loads_on_cpu_runtime():
    loaded = check_hf_sentiment_pipeline()
    assert loaded["sentiment_pipeline"] == "TextClassificationPipeline"
    assert loaded["sentiment_tokenizer"]
    assert loaded["sentiment_model"]


def test_embedding_gemma_encoder_loads_on_cpu_runtime():
    assert check_embedding_gemma_model() == "SentenceTransformer"


def test_gemma_lm_loads_or_is_explicitly_disabled():
    if os.getenv("OPENWILLIS_DISABLE_PERPLEXITY_TESTS") or os.getenv("OPENWILLIS_ALLOW_MISSING_GEMMA"):
        pytest.skip("Gemma LM readiness disabled by explicit test-mode environment flag")

    loaded = check_gemma_causal_lm()
    assert loaded["causal_tokenizer"]
    assert loaded["causal_model"]
