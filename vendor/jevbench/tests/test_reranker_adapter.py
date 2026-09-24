import math
from types import SimpleNamespace

from jevbench.adapters.reranker import (
    apply_noul_threshold,
    calibrated_probs,
    fit_noul_threshold,
    fit_temperature,
    option_documents,
    query_text,
    softmax,
    task_instruction,
)


def task(kind="choice", expected="b"):
    criteria = {"a": "first", "b": "second"}
    labels = ["a", "b"]
    if kind == "noul":
        criteria, labels = {"false": "condition absent", "true": "condition present"}, ["no", "yes"]
    if kind == "score":
        criteria, labels, expected = ["low", "high"], ["0", "1"], 1
    return SimpleNamespace(state={"z": 2, "a": 1}, question={"type": kind, "instructions": "Decide.", "criteria": criteria}, labels=labels, expected=expected)


def test_neutral_rendering_is_complete_and_stable():
    t = task()
    assert query_text(t).startswith('Situation:\n{"a":1,"z":2}')
    assert "Question:\nDecide." in query_text(t)
    assert option_documents(t) == ["Answer option a: first", "Answer option b: second"]
    assert task_instruction(t).startswith("Classify:")


def test_softmax_and_threshold_are_distributions():
    assert softmax([3, 3], 1) == [0.5, 0.5]
    shifted = apply_noul_threshold({"no": 0.6, "yes": 0.4}, 0.4)
    assert math.isclose(sum(shifted.values()), 1)
    assert math.isclose(shifted["yes"], 0.5)
    probs = calibrated_probs(task("noul", "yes"), [-1, 1], 2, 0.5)
    assert probs["yes"] > probs["no"]


def test_public_grid_fit_is_deterministic():
    rows = [(task(expected="b"), [0, 4]), (task(expected="a"), [4, 0])]
    temperature, curve = fit_temperature(rows)
    assert temperature == 0.25
    assert len(curve) == 6
    nouls = [(task("noul", "yes"), [0, 2]), (task("noul", "no"), [2, 0])]
    threshold, threshold_curve = fit_noul_threshold(nouls, 1)
    assert threshold in {row["threshold"] for row in threshold_curve}
    assert len(threshold_curve) == 7
