#!/usr/bin/env python3
"""Track A Ukrainian POS tagging and verb-tense validation.

This script converts `notebooks/pos_tense.ipynb` into a reusable command that
reads Universal Dependencies CoNLL-U files, runs spaCy and Stanza Ukrainian POS
pipelines, aligns predicted tokens back to UD token spans, and writes aggregate
tables corresponding to manuscript Tables 3-5.
"""

from __future__ import annotations

import argparse
import importlib
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd


UPOS10 = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "NOUN", "PRON", "PROPN", "VERB"]


def require_module(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"Missing optional dependency '{module_name}'. Install {install_hint}.") from exc


def rebuild_text_and_offsets(tokens: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        form = token["form"]
        previous_no_space = bool(parts and spans and token.get("_prev_no_space", False))
        if parts and not previous_no_space:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(form)
        cursor += len(form)
        spans.append((start, cursor))
        token["_this_no_space"] = (token.get("misc") or {}).get("SpaceAfter") == "No"
    for idx in range(1, len(tokens)):
        tokens[idx]["_prev_no_space"] = tokens[idx - 1].get("_this_no_space", False)
    return "".join(parts), spans


def open_conllu(path_or_url: str):
    if path_or_url.startswith(("http://", "https://")):
        requests = require_module("requests", "paper/track-a-requirements.txt")
        response = requests.get(path_or_url, timeout=60)
        response.raise_for_status()
        return StringIO(response.text)
    return open(path_or_url, encoding="utf-8")


def feats_to_dict(feats) -> dict[str, str]:
    if feats is None:
        return {}
    if isinstance(feats, dict):
        return feats
    if isinstance(feats, str):
        return dict(kv.split("=", 1) for kv in feats.split("|") if "=" in kv)
    return {}


def read_ud_conllu(path_or_url: str) -> pd.DataFrame:
    conllu = require_module("conllu", "paper/track-a-requirements.txt")
    rows = []
    with open_conllu(path_or_url) as handle:
        for sent in conllu.parse_incr(handle):
            sent_id = (sent.metadata or {}).get("sent_id") or (sent.metadata or {}).get("sentid") or ""
            ud_tokens = [token for token in sent if isinstance(token["id"], int)]
            text_from_meta = (sent.metadata or {}).get("text")
            if text_from_meta:
                text = text_from_meta
                spans = []
                cursor = 0
                for token in ud_tokens:
                    form = token["form"]
                    pos = text.find(form, cursor)
                    if pos < 0:
                        pos = cursor
                    spans.append((pos, pos + len(form)))
                    cursor = pos + len(form)
            else:
                text, spans = rebuild_text_and_offsets(ud_tokens)

            for token, (start, end) in zip(ud_tokens, spans):
                feats = feats_to_dict(token.get("feats"))
                rows.append(
                    {
                        "sent_id": sent_id,
                        "text": text,
                        "token_id": token["id"],
                        "form": token["form"],
                        "lemma": token.get("lemma"),
                        "gold_upos": token.get("upos"),
                        "gold_xpos": token.get("xpos"),
                        "gold_feats": feats,
                        "gold_tense": feats.get("Tense"),
                        "head": token.get("head"),
                        "deprel": token.get("deprel"),
                        "space_after": not ((token.get("misc") or {}).get("SpaceAfter") == "No"),
                        "char_start": start,
                        "char_end": end,
                        "spacy_upos": None,
                        "spacy_tense": None,
                        "stanza_upos": None,
                        "stanza_tense": None,
                    }
                )
    df = pd.DataFrame(rows)
    for column in ["gold_upos", "gold_tense"]:
        df[column] = df[column].astype("category")
    return df


def macro_f1(df: pd.DataFrame, true_col: str, pred_col: str, labels: Iterable[str] | None = None) -> float:
    metrics = require_module("sklearn.metrics", "paper/requirements.txt")
    y_true = df[true_col].astype(str)
    y_pred = df[pred_col].astype(str)
    if labels is None:
        labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
    return float(metrics.f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def macro_f1_tense(df: pd.DataFrame, pred_col: str, restrict_to_verbs: bool = True) -> float:
    metrics = require_module("sklearn.metrics", "paper/requirements.txt")
    data = df[df["gold_upos"] == "VERB"] if restrict_to_verbs else df

    def norm(series: pd.Series) -> pd.Series:
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.cat.add_categories(["None"]).fillna("None")
        else:
            series = series.fillna("None").astype(str)
        return series.astype(str)

    y_true = norm(data["gold_tense"])
    y_pred = norm(data[pred_col])
    labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
    return float(metrics.f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def span_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def align_by_spans(ud_spans: list[tuple[int, int]], pred_spans: list[tuple[int, int]]) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = defaultdict(list)
    for i, ud_span in enumerate(ud_spans):
        for j, pred_span in enumerate(pred_spans):
            if span_overlap(ud_span, pred_span) > 0:
                mapping[i].append(j)
    return mapping


def run_spacy_and_attach(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    spacy = require_module("spacy", "paper/track-a-requirements.txt")
    nlp = spacy.load(model_name, disable=[])
    out = df.copy()
    for _, seg in out.groupby("sent_id", sort=False):
        text = seg["text"].iloc[0]
        doc = nlp(text)
        pred_spans = [(token.idx, token.idx + len(token.text)) for token in doc]
        pred_upos = [token.pos_ for token in doc]
        pred_tense = [token.morph.get("Tense", [None])[0] if token.morph.get("Tense") else None for token in doc]
        mapping = align_by_spans(list(zip(seg["char_start"].tolist(), seg["char_end"].tolist())), pred_spans)
        agg_upos, agg_tense = [], []
        for i_ud in range(len(seg)):
            js = mapping.get(i_ud, [])
            if not js:
                agg_upos.append(None)
                agg_tense.append(None)
                continue
            upos = Counter(pred_upos[j] for j in js if pred_upos[j] is not None).most_common(1)
            tenses = [pred_tense[j] for j in js if pred_tense[j] not in (None, "")]
            agg_upos.append(upos[0][0] if upos else None)
            agg_tense.append(Counter(tenses).most_common(1)[0][0] if tenses else None)
        out.loc[seg.index, "spacy_upos"] = agg_upos
        out.loc[seg.index, "spacy_tense"] = agg_tense
    out["spacy_upos"] = out["spacy_upos"].astype("category")
    out["spacy_tense"] = out["spacy_tense"].astype("category")
    return out


def run_stanza_and_attach(df: pd.DataFrame, *, download: bool, processors: str) -> pd.DataFrame:
    stanza = require_module("stanza", "paper/track-a-requirements.txt")
    if download:
        stanza.download("uk")
    nlp = stanza.Pipeline("uk", processors=processors)
    out = df.copy()
    for _, seg in out.groupby("sent_id", sort=False):
        text = seg["text"].iloc[0]
        doc = nlp(text)
        pred_spans, pred_upos, pred_tense = [], [], []
        for sent in doc.sentences:
            for token in sent.tokens:
                start, end = token.start_char, token.end_char
                if start is None or end is None:
                    continue
                upos_list = [word.upos for word in token.words if word.upos]
                feats_dicts = [feats_to_dict(word.feats) for word in token.words if word.feats]
                tenses = [feats.get("Tense") for feats in feats_dicts if feats.get("Tense")]
                pred_spans.append((start, end))
                pred_upos.append(Counter(upos_list).most_common(1)[0][0] if upos_list else None)
                pred_tense.append(Counter(tenses).most_common(1)[0][0] if tenses else None)
        mapping = align_by_spans(list(zip(seg["char_start"].tolist(), seg["char_end"].tolist())), pred_spans)
        agg_upos, agg_tense = [], []
        for i_ud in range(len(seg)):
            js = mapping.get(i_ud, [])
            if not js:
                agg_upos.append(None)
                agg_tense.append(None)
                continue
            upos = Counter(pred_upos[j] for j in js if pred_upos[j] is not None).most_common(1)
            tenses = [pred_tense[j] for j in js if pred_tense[j] not in (None, "")]
            agg_upos.append(upos[0][0] if upos else None)
            agg_tense.append(Counter(tenses).most_common(1)[0][0] if tenses else None)
        out.loc[seg.index, "stanza_upos"] = agg_upos
        out.loc[seg.index, "stanza_tense"] = agg_tense
    out["stanza_upos"] = out["stanza_upos"].astype("category")
    out["stanza_tense"] = out["stanza_tense"].astype("category")
    return out


def validate_ud_df(df: pd.DataFrame) -> None:
    required = {"sent_id", "text", "token_id", "gold_upos", "char_start", "char_end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"UD DataFrame missing columns: {sorted(missing)}")
    duplicate_count = df.duplicated(["sent_id", "token_id"]).sum()
    if duplicate_count:
        raise ValueError(f"Duplicate (sent_id, token_id) rows: {duplicate_count}")
    bad_spans = ((df["char_end"] <= df["char_start"]) | (df["char_start"] < 0)).sum()
    if bad_spans:
        raise ValueError(f"Invalid spans: {bad_spans}")
    empty_text = (df.groupby("sent_id")["text"].first().str.len() == 0).sum()
    if empty_text:
        raise ValueError(f"Sentences with empty text: {empty_text}")


def parse_dataset(value: str) -> tuple[str, str]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), path.strip()
    path = Path(value)
    return path.stem, value


def compute_summary(dataset_name: str, df: pd.DataFrame) -> tuple[dict, dict, dict]:
    pos = {
        "dataset": dataset_name,
        "spacy_upos_macro_f1": macro_f1(df, "gold_upos", "spacy_upos", labels=UPOS10) if "spacy_upos" in df else float("nan"),
        "stanza_upos_macro_f1": macro_f1(df, "gold_upos", "stanza_upos", labels=UPOS10) if "stanza_upos" in df else float("nan"),
    }
    pos["delta_f1_stanza_minus_spacy_pct"] = 100 * (pos["stanza_upos_macro_f1"] - pos["spacy_upos_macro_f1"])

    tense_verbs = {
        "dataset": dataset_name,
        "spacy_tense_macro_f1": macro_f1_tense(df, "spacy_tense", restrict_to_verbs=True) if "spacy_tense" in df else float("nan"),
        "stanza_tense_macro_f1": macro_f1_tense(df, "stanza_tense", restrict_to_verbs=True) if "stanza_tense" in df else float("nan"),
    }
    tense_verbs["delta_f1_stanza_minus_spacy_pct"] = 100 * (
        tense_verbs["stanza_tense_macro_f1"] - tense_verbs["spacy_tense_macro_f1"]
    )

    has_tense = df[df["gold_tense"].notna()]
    tense_all = {
        "dataset": dataset_name,
        "spacy_tense_macro_f1": macro_f1_tense(has_tense, "spacy_tense", restrict_to_verbs=False) if "spacy_tense" in df else float("nan"),
        "stanza_tense_macro_f1": macro_f1_tense(has_tense, "stanza_tense", restrict_to_verbs=False) if "stanza_tense" in df else float("nan"),
    }
    tense_all["delta_f1_stanza_minus_spacy_pct"] = 100 * (
        tense_all["stanza_tense_macro_f1"] - tense_all["spacy_tense_macro_f1"]
    )
    return pos, tense_verbs, tense_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset as name=path_or_url. Repeat for train/dev/test.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--spacy-model", default="uk_core_news_md")
    parser.add_argument("--skip-spacy", action="store_true")
    parser.add_argument("--skip-stanza", action="store_true")
    parser.add_argument("--stanza-download", action="store_true")
    parser.add_argument("--stanza-processors", default="tokenize,mwt,pos,lemma")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_frames = []
    pos_rows, tense_verb_rows, tense_has_tense_rows = [], [], []
    for dataset_arg in args.dataset:
        dataset_name, path = parse_dataset(dataset_arg)
        df = read_ud_conllu(path)
        validate_ud_df(df)
        if not args.skip_spacy:
            df = run_spacy_and_attach(df, args.spacy_model)
        if not args.skip_stanza:
            df = run_stanza_and_attach(df, download=args.stanza_download, processors=args.stanza_processors)
        df.insert(0, "dataset", dataset_name)
        token_frames.append(df)
        pos, tense_verbs, tense_all = compute_summary(dataset_name, df)
        pos_rows.append(pos)
        tense_verb_rows.append(tense_verbs)
        tense_has_tense_rows.append(tense_all)

    pd.concat(token_frames, ignore_index=True).to_csv(args.output_dir / "pos_tense_token_predictions.csv", index=False)
    pd.DataFrame(pos_rows).to_csv(args.output_dir / "table03_pos_tagging_upos_macro_f1.csv", index=False)
    pd.DataFrame(tense_verb_rows).to_csv(args.output_dir / "table04_tense_prediction_verb_only.csv", index=False)
    pd.DataFrame(tense_has_tense_rows).to_csv(args.output_dir / "table05_tense_prediction_has_gold_tense.csv", index=False)


if __name__ == "__main__":
    main()
