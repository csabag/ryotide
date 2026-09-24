"""Small CLI over the two modes. `uv run python -m ryotide.cli --help`"""
from __future__ import annotations
import argparse, json, sys

from .model import DEFAULT_MODEL, load_model, describe
from .classify import Classifier, LabelSet
from .branch import BranchedClassifier, projected_branch_memory, cache_bytes
from .generate import classify_then_generate


def main(argv=None):
    p = argparse.ArgumentParser(prog="ryotide")
    p.add_argument("--model", default=DEFAULT_MODEL)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify", help="classify questions against a state file")
    c.add_argument("state", help="path to a file holding the shared state ('-' for stdin)")
    c.add_argument("-q", "--question", action="append", required=True)
    c.add_argument("--labels", nargs="+", default=["yes", "no"])
    c.add_argument("--strategy", default="lazy", choices=["lazy", "trim", "fork"])

    g = sub.add_parser("hybrid", help="classify then continue generating from that branch")
    g.add_argument("state")
    g.add_argument("-q", "--question", required=True)
    g.add_argument("-f", "--follow-up", required=True)
    g.add_argument("--labels", nargs="+", default=["yes", "no"])
    g.add_argument("--format", default="json", choices=["json", "prose", "bare"])
    g.add_argument("--max-tokens", type=int, default=160)

    m = sub.add_parser("memory", help="KV budget for N branches, no model load needed")
    m.add_argument("--state-tokens", type=int, default=6000)
    m.add_argument("--n", type=int, default=10)

    a = p.parse_args(argv)

    if a.cmd == "memory":
        model, _ = load_model(a.model)
        info = describe(model, a.model)
        print(json.dumps(projected_branch_memory(info.kv_bytes_per_token, a.state_tokens, a.n), indent=1))
        return

    state = sys.stdin.read() if a.state == "-" else open(a.state).read()
    model, tok = load_model(a.model)
    clf = Classifier(model, tok)
    labels = LabelSet.resolve(tok, a.labels)
    bc = BranchedClassifier(clf)
    n = bc.set_state(state)
    print(f"state: {n} tokens, KV {cache_bytes(bc.state_cache)/1e6:.1f}MB", file=sys.stderr)

    if a.cmd == "classify":
        for r in bc.classify_many(a.question, labels, strategy=a.strategy):
            print(json.dumps({"question": r.question, **r.decision.as_json()}))
    else:
        res = bc.classify_many([a.question], labels, strategy="lazy")[0]
        branch = bc.materialize(a.question)
        out = classify_then_generate(clf, branch, res.decision, a.follow_up,
                                     fmt=a.format, max_tokens=a.max_tokens)
        print(json.dumps({"classification": res.decision.as_json(),
                          "generation": out.text,
                          "decode_tok_per_s": round(out.decode_tps, 1)}, indent=1))


if __name__ == "__main__":
    main()
