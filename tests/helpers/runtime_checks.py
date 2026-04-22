from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProbe:
    name: str
    value: str


def check_package_versions() -> list[RuntimeProbe]:
    from tests.helpers.pipeline_runner import _speech_characteristics

    _speech_characteristics()

    import huggingface_hub
    import nltk
    import sentencepiece
    import sentence_transformers
    import spacy
    import torch
    import transformers

    probes = [
        RuntimeProbe("openwillis_speech_module", "loaded"),
        RuntimeProbe("huggingface_hub", huggingface_hub.__version__),
        RuntimeProbe("nltk", nltk.__version__),
        RuntimeProbe("sentencepiece", sentencepiece.__version__),
        RuntimeProbe("sentence_transformers", sentence_transformers.__version__),
        RuntimeProbe("spacy", spacy.__version__),
        RuntimeProbe("torch", torch.__version__),
        RuntimeProbe("transformers", transformers.__version__),
    ]
    return probes


def check_spacy_models() -> dict[str, str]:
    import spacy

    loaded = {}
    for model_name in ("en_core_web_sm", "uk_core_news_sm"):
        loaded[model_name] = type(spacy.load(model_name)).__name__
    return loaded


def check_nltk_resources() -> list[str]:
    import nltk

    required = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    found = []
    for resource_path, label in required:
        nltk.data.find(resource_path)
        found.append(label)
    return found


def check_hf_sentiment_pipeline() -> dict[str, str]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=-1,
    )

    return {
        "sentiment_pipeline": type(sentiment_pipeline).__name__,
        "sentiment_tokenizer": type(tokenizer).__name__,
        "sentiment_model": type(model).__name__,
    }


def check_embedding_gemma_model() -> str:
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("google/embeddinggemma-300m", device="cpu")
    return type(embedding_model).__name__


def check_gemma_causal_lm() -> dict[str, str]:
    import torch
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to initialize google/gemma-3-270m")

    login(token=token, add_to_git_credential=False)

    causal_model_id = "google/gemma-3-270m"
    causal_tokenizer = AutoTokenizer.from_pretrained(causal_model_id)
    causal_model = AutoModelForCausalLM.from_pretrained(causal_model_id)
    causal_model.to(torch.device("cpu"))
    causal_model.eval()

    return {
        "causal_tokenizer": type(causal_tokenizer).__name__,
        "causal_model": type(causal_model).__name__,
    }


def check_huggingface_models() -> dict[str, str]:
    import torch
    from huggingface_hub import login
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to initialize gated Hugging Face models")
    login(token=token, add_to_git_credential=False)

    sentiment_model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_id, use_fast=True)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=sentiment_model_id,
        tokenizer=sentiment_tokenizer,
        top_k=None,
        device=-1,
    )

    embedding_model = SentenceTransformer("google/embeddinggemma-300m", device="cpu")

    causal_model_id = "google/gemma-3-270m"
    causal_tokenizer = AutoTokenizer.from_pretrained(causal_model_id)
    causal_model = AutoModelForCausalLM.from_pretrained(causal_model_id)
    causal_model.to(torch.device("cpu"))
    causal_model.eval()

    return {
        "sentiment_pipeline": type(sentiment_pipeline).__name__,
        "sentiment_tokenizer": type(sentiment_tokenizer).__name__,
        "embedding_model": type(embedding_model).__name__,
        "causal_tokenizer": type(causal_tokenizer).__name__,
        "causal_model": type(causal_model).__name__,
    }
