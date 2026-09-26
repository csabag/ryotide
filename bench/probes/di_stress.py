"""Decision Index request shapes against a running server: does every shape get an answer?

Not a score. The suite rows are private, so these are synthetic requests built to the
published shapes (decision-index docs/format.md and the per-benchmark request/field
counts): question counts, menu sizes, object instructions, JSON and empty states,
noul questions, long documents. Pass = HTTP 200 with a valid answer for every question.

    uv run python bench/probes/di_stress.py http://127.0.0.1:8000
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/") + "/v1/systemone"
rng = random.Random(0)
WORDS = ("the account model data service report user system value process order time result "
         "policy party agreement information within shall may any other such provided notice "
         "request response query tool function return string number list city date price").split()


def text(n_words):
    return " ".join(rng.choice(WORDS) for _ in range(n_words)).capitalize() + "."


def choice(instr, labels, crit=True):
    return {"type": "choice", "instructions": instr,
            "criteria": {l: (f"Option {l.replace('_', ' ')}." if crit else "") for l in labels}}


YN = {"yes": "Relevant/useful to the query.", "no": "Not relevant/useful to the query."}


def retrieval(n, doc_words):
    return {f"c{i:03d}": {"type": "choice", "criteria": YN, "instructions": {
        "task": "Assess whether this candidate is relevant/useful to the query. Use the full text "
                "below; return the probability of relevance.", "candidate": text(doc_words)}}
        for i in range(n)}


CASES = [
    ("ToolRet: 200 questions, object instructions", {"state": "Query: " + text(30), "questions": retrieval(200, 120)}),
    ("BRIGHT: 70 questions, long documents", {"state": "Question: " + text(120), "questions": retrieval(70, 700)}),
    ("ACOS: 58 yes/no aspect questions", {"state": "Review: " + text(150), "questions": {
        f"p{i}": choice(f"Does the review express aspect {i} with this sentiment?", ["no", "yes"]) for i in range(58)}}),
    ("ContractNLI: ~9k-word NDA, 17 questions", {"state": text(9000), "questions": {
        f"h{i}": choice(f"Statement {i}: " + text(20), ["contradiction", "entailment", "not_mentioned"]) for i in range(17)}}),
    ("MuSR: long story, 1 question", {"state": text(1500), "questions": {
        "q": choice("Who is the most likely murderer?", ["Alice", "Bob"])}}),
    ("POP909: 129 options", {"state": "Notes: C4 E4 G4 B3 | D4 F4 A4", "questions": {
        "q": choice("Which chord is playing on this beat?", [f"chord_{i}" for i in range(129)])}}),
    ("255 options (API cap)", {"state": text(40), "questions": {
        "q": choice("Pick the matching entry.", [f"item_{i}" for i in range(255)])}}),
    ("CLINC-style 151 options, no criteria text", {"state": "how do i set an alarm", "questions": {
        "q": choice("Intent?", [f"intent_{i}" for i in range(150)] + ["oos"], crit=False)}}),
    ("Home appliance: JSON state, 18 questions", {"state": {"rooms": {"kitchen": {"oven": "off", "temp_c": 21}},
        "command": "preheat the oven to 200"}, "questions": {
        f"f{i}": choice(f"Field {i}: how should this be handled?", ["allow", "deny", "ask"]) for i in range(18)}}),
    ("RAGTruth: noul with true/false criteria", {"state": {"prompt": text(200), "response": text(80)},
        "questions": {"q": {"type": "noul", "instructions": "The response contains content that is not "
                            "supported by the context in the prompt.",
                            "criteria": {"true": "Unsupported content.", "false": "All supported."}}}}),
    ("Empty state", {"state": "", "questions": {"q": choice("Is 7 a prime number?", ["no", "yes"])}}),
    ("~30k-token state, 3 questions", {"state": text(24000), "questions": {
        f"q{i}": choice(f"Question {i}?", ["a", "b", "c", "d"]) for i in range(3)}}),
]

fails = 0
for name, body in CASES:
    req = urllib.request.Request(URL, json.dumps({"model": "default", **body}).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.load(r)
        dt = time.perf_counter() - t0
        ans = out["answers"]
        good = all(k in ans for k in body["questions"]) and all(
            (a["type"] == "noul" and 0 <= a["noul"] <= 1) or
            (a["type"] == "choice" and a["choice"] in body["questions"][k]["criteria"]
             and abs(sum(a["probabilities"].values()) - 1) < 1e-3) for k, a in ans.items())
        fails += not good
        rt = out.get("runtime", {})
        print(f"{'OK ' if good else 'BAD'} {name:44s} {dt:7.1f}s  q={len(ans):3d}  in_tok={out['usage']['input_tokens']:>8,}  "
              f"shared={rt.get('shared_prefix_tokens')}  min_mass={rt.get('marker_mass', 0):.3f}", flush=True)
    except urllib.error.HTTPError as e:
        fails += 1
        print(f"ERR {name:44s} HTTP {e.code}: {e.read()[:200]!r}", flush=True)
    except Exception as e:
        fails += 1
        print(f"ERR {name:44s} {type(e).__name__}: {e}", flush=True)
print(f"\n{len(CASES) - fails}/{len(CASES)} shapes answered")
