"""GLUE accuracy harness -- spec step 3's gate.

"Run against a GLUE subset and get a real accuracy number. This has not been
done. Nothing past this point matters if accuracy is bad."

Every task is posed as the same primitive: build a prompt, read logits at the
final position, mask to that task's label tokens. No decoding anywhere.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import mlx.core as mx

from .classify import Classifier, LabelSet


@dataclass
class GlueTask:
    name: str
    config: str
    labels: list[str]              # index i == dataset label int i
    template: Callable[[dict], str]
    content_free: Callable[[str], str]
    split: str = "validation"


def _t(s: str) -> str:
    return s.strip()


TASKS: dict[str, GlueTask] = {
    "sst2": GlueTask(
        name="sst2",
        config="sst2",
        labels=["negative", "positive"],
        template=lambda e: f'Review: "{_t(e["sentence"])}"\nIs the sentiment of this review positive or negative?\nAnswer:',
        content_free=lambda cf: f'Review: "{cf}"\nIs the sentiment of this review positive or negative?\nAnswer:',
    ),
    "rte": GlueTask(
        name="rte",
        config="rte",
        labels=["yes", "no"],      # 0=entailment, 1=not_entailment
        template=lambda e: f'Premise: "{_t(e["sentence1"])}"\nHypothesis: "{_t(e["sentence2"])}"\nDoes the premise entail the hypothesis? Answer yes or no.\nAnswer:',
        content_free=lambda cf: f'Premise: "{cf}"\nHypothesis: "{cf}"\nDoes the premise entail the hypothesis? Answer yes or no.\nAnswer:',
    ),
    "mrpc": GlueTask(
        name="mrpc",
        config="mrpc",
        labels=["no", "yes"],      # 0=not_equivalent, 1=equivalent
        template=lambda e: f'Sentence 1: "{_t(e["sentence1"])}"\nSentence 2: "{_t(e["sentence2"])}"\nDo these two sentences mean the same thing? Answer yes or no.\nAnswer:',
        content_free=lambda cf: f'Sentence 1: "{cf}"\nSentence 2: "{cf}"\nDo these two sentences mean the same thing? Answer yes or no.\nAnswer:',
    ),
    "qnli": GlueTask(
        name="qnli",
        config="qnli",
        labels=["yes", "no"],      # 0=entailment, 1=not_entailment
        template=lambda e: f'Question: {_t(e["question"])}\nSentence: "{_t(e["sentence"])}"\nDoes the sentence contain the answer to the question? Answer yes or no.\nAnswer:',
        content_free=lambda cf: f'Question: {cf}\nSentence: "{cf}"\nDoes the sentence contain the answer to the question? Answer yes or no.\nAnswer:',
    ),
    "cola": GlueTask(
        name="cola",
        config="cola",
        labels=["no", "yes"],      # 0=unacceptable, 1=acceptable
        template=lambda e: f'Sentence: "{_t(e["sentence"])}"\nIs this sentence grammatically acceptable English? Answer yes or no.\nAnswer:',
        content_free=lambda cf: f'Sentence: "{cf}"\nIs this sentence grammatically acceptable English? Answer yes or no.\nAnswer:',
    ),
    "mnli": GlueTask(
        name="mnli",
        config="mnli",
        labels=["yes", "maybe", "no"],   # 0=entailment, 1=neutral, 2=contradiction
        template=lambda e: f'Premise: "{_t(e["premise"])}"\nHypothesis: "{_t(e["hypothesis"])}"\nIs the hypothesis true given the premise? Answer yes, maybe, or no.\nAnswer:',
        content_free=lambda cf: f'Premise: "{cf}"\nHypothesis: "{cf}"\nIs the hypothesis true given the premise? Answer yes, maybe, or no.\nAnswer:',
        split="validation_matched",
    ),
}


def matthews_corr(y_true: list[int], y_pred: list[int]) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denom == 0 else (tp * tn - fp * fn) / denom


def f1(y_true: list[int], y_pred: list[int], positive: int = 1) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def build_prompt(clf: Classifier, text: str, chat: bool) -> list[int]:
    if not chat:
        return clf.encode(text)
    msgs = [{"role": "user", "content": text}]
    return clf.tokenizer.apply_chat_template(msgs, add_generation_prompt=True)


def evaluate(
    clf: Classifier,
    task: GlueTask,
    n: int = 200,
    chat: bool = True,
    calibrate: bool = True,
    seed: int = 0,
    progress: bool = False,
) -> dict:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", task.config, split=task.split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n]

    labels = LabelSet.resolve(clf.tokenizer, task.labels)

    bias = None
    if calibrate:
        rows = []
        for cf in ("N/A", "", "[MASK]"):
            toks = build_prompt(clf, task.content_free(cf), chat)
            cache = clf.prefill([])
            row = clf.logits_at_end(cache, toks)
            sel = row[mx.array(labels.token_ids)].astype(mx.float32)
            rows.append(mx.log(mx.softmax(sel)))
        bias = mx.mean(mx.stack(rows), axis=0)

    y_true: list[int] = []
    y_pred: list[int] = []
    margins: list[float] = []
    t0 = time.time()
    total_tokens = 0
    for k, i in enumerate(idx):
        ex = ds[i]
        toks = build_prompt(clf, task.template(ex), chat)
        total_tokens += len(toks)
        cache = clf.prefill([])
        d = clf.classify(cache, toks, labels, bias=bias)
        y_true.append(int(ex["label"]))
        y_pred.append(d.index)
        margins.append(d.margin)
        del cache
        if progress and (k + 1) % 50 == 0:
            print(f"  {task.name}: {k+1}/{len(idx)}", flush=True)

    elapsed = time.time() - t0
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    majority = max(y_true.count(c) for c in set(y_true)) / len(y_true)
    out = {
        "task": task.name,
        "n": len(y_true),
        "chat": chat,
        "calibrated": calibrate,
        "accuracy": round(acc, 4),
        "majority_baseline": round(majority, 4),
        "mean_margin": round(sum(margins) / len(margins), 4),
        "pred_distribution": {task.labels[c]: y_pred.count(c) for c in range(len(task.labels))},
        "seconds": round(elapsed, 2),
        "ms_per_example": round(1000 * elapsed / len(y_true), 2),
        "mean_prompt_tokens": round(total_tokens / len(y_true), 1),
    }
    if task.name == "cola":
        out["matthews"] = round(matthews_corr(y_true, y_pred), 4)
    if task.name == "mrpc":
        out["f1"] = round(f1(y_true, y_pred), 4)
    return out
