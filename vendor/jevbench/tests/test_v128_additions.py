"""v1.2.8: the requested entrants measured in add-requests run 4, all complete and ranked."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = json.loads((ROOT / "results/v1.2/jevbench-v1.2-results.json").read_text())
SYS = {s["key"]: s for s in ART["systems"]}
GPU_ROWS = ["decider-35b-a3b", "reflex-27b", "jqv", "decider-2b", "reflex-4b", "opendecision"]
API_ROWS = ["decision-machine-1"]
CPU_ROWS = ["gliner2-large"]
OPTIONAL = ["jev-local", "nimble-9b"]   # present only if their runs completed
NEW = GPU_ROWS + API_ROWS + CPU_ROWS + [k for k in OPTIONAL if k in SYS]


def test_v128_rows_are_complete_and_ranked():
    assert tuple(map(int, ART["revision"].removeprefix("v").split("."))) >= (1, 2, 8)
    for key in NEW:
        row = SYS[key]
        assert row["listing"] == "ranked" and row["rank"], key
        assert not row["partial"], key
        assert row["hard"]["coverage"] == 1.0, key
        assert row["licence"], key
        assert all(row["tiers"][t] is not None for t in ("easy", "standard", "judge", "hard")), key


def test_endpoint_kinds_match_where_each_system_ran():
    for key in GPU_ROWS:
        assert SYS[key]["endpoint_kind"] == "gpu", key
    for key in API_ROWS:
        assert SYS[key]["endpoint_kind"] == "api", key
        assert SYS[key]["speed"]["p50_s_adjusted"] == SYS[key]["speed"]["p50_s_raw"], key
    for key in CPU_ROWS:
        assert SYS[key]["endpoint_kind"] == "cpu", key


def test_no_new_row_gets_a_free_cost_score():
    for key in NEW:
        assert SYS[key]["cost"]["usd_per_1000"] > 0, key
        assert SYS[key]["axes"]["cost"] < 100, key


def test_decision_machine_is_priced_at_its_public_tariff():
    c = SYS["decision-machine-1"]["cost"]
    assert c["kind"] == "measured" and "$0.04 per million input tokens" in c["basis"]


def test_nimble_rerun_replaces_the_old_row_and_keeps_its_old_measurement():
    old = ART["superseded_rows"]["nimble-9b"]
    assert old["old_tiers"]["hard"] < SYS["nimble-9b"]["tiers"]["hard"]
    assert SYS["nimble-9b"]["hard"]["coverage"] == 1.0 and "PR #4" in old["reason"]


def test_certo_v129_is_complete_ranked_and_costed_from_retained_tokens():
    row = SYS["certo"]
    assert row["listing"] == "ranked" and row["rank"]
    assert row["hard"]["coverage"] == 1.0 and not row["partial"]
    assert row["endpoint_kind"] == "gpu"
    assert row["cost"]["usd_per_1000"] > 0
    assert "input tokens measured" in row["cost"]["basis"]


def test_smalljev_v1210_is_complete_ranked_and_not_costed_as_free():
    row = SYS["smalljev"]
    assert row["listing"] == "ranked" and row["rank"]
    assert row["hard"]["coverage"] == 1.0 and not row["partial"]
    assert row["endpoint_kind"] == "gpu"
    assert row["licence"] == "Apache-2.0"
    assert row["cost"]["usd_per_1000"] > 0 and row["axes"]["cost"] < 100
    assert "input tokens measured" in row["cost"]["basis"]
