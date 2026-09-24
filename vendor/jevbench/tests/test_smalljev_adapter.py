from jevbench.adapters import SmallJevLocalAdapter
from jevbench.tasks import Task


def task(qtype, criteria, labels):
    return Task(id="t", family="f", state={"z": 2, "a": 1}, labels=labels,
                expected=labels[0], split="public",
                question={"type": qtype, "instructions": "Decide this.",
                          "criteria": criteria})


def test_smalljev_choice_mapping_is_stable_and_excludes_gold():
    row = SmallJevLocalAdapter.build_request(task(
        "choice", {"a": "Alpha", "b": "Beta"}, ["a", "b"]))
    assert row["state"] == '{"a": 1, "z": 2}'
    assert row["labels"] == ["a", "b"]
    assert row["options"] == ["a", "b"]
    assert row["rubric"] == {"a": "Alpha", "b": "Beta"}
    assert "expected" not in row and "gold" not in row


def test_smalljev_noul_and_score_mapping():
    row = SmallJevLocalAdapter.build_request(task(
        "noul", {"true": "T", "false": "F"}, ["no", "yes"]))
    assert row["options"] == ["no", "yes"]
    assert row["rubric"] == {"no": "F", "yes": "T"}
    row = SmallJevLocalAdapter.build_request(task(
        "score", ["low", "mid", "high"], ["0", "1", "2"]))
    assert row["options"] == ["low", "mid", "high"]
