"""Where does the 25ms per question go? Fixed per-forward-pass cost, or
attention over the 6K prefix? The answer decides whether masked batched
attention (deferred in the spec) would actually buy anything."""
import sys, time; sys.path.insert(0,"src")
import mlx.core as mx
from ryotide.model import load_model
from ryotide.classify import Classifier, LabelSet
from ryotide.branch import BranchedClassifier, fork_cache

model, tok = load_model(); clf = Classifier(model, tok); bc = BranchedClassifier(clf)
labels = LabelSet.resolve(tok, ["yes","no"])
para = "Incident 4471: the ingest worker stalled after the queue backed up. No data loss. "

def timed(fn, reps=8):
    fn(); fn()
    t=time.time()
    for _ in range(reps): fn()
    return (time.time()-t)/reps*1000

import os
if os.environ.get("SKIP_A"): print("A) skipped")
print("A) question length at fixed 6K state  -- tests attention-over-prefix cost")
bc.set_state(para * 400)
base = bc.state_len
for qlen in (1, 17, 64, 256, 1024):
    qt = clf.encode(" ping" * qlen)[:qlen]
    ms = timed(lambda: clf.classify(fork_cache(bc.state_cache), qt, labels))
    print(f"   q={qlen:5d} tok -> {ms:6.1f}ms  ({ms/qlen:.2f} ms/tok)")

print("\nB) state length at fixed 17-tok question -- tests prefix scaling")
qt = clf.encode("\nQ: Was there data loss? Answer yes or no.\nA:")
for target in (0, 1000, 3000, 6000, 12000):
    b2 = BranchedClassifier(clf)
    b2.set_state(para * max(1, target // 20) if target else "")
    ms = timed(lambda: clf.classify(fork_cache(b2.state_cache), qt, labels))
    print(f"   state={b2.state_len:6d} tok -> {ms:6.1f}ms")
    del b2

print("\nC) batched questions in ONE forward pass (no shared-prefix mask, just batch)")
b3 = BranchedClassifier(clf); b3.set_state(para*400)
qt = clf.encode("\nQ: Was there data loss? Answer yes or no.\nA:")
for n in (1, 5, 10, 25):
    def run(n=n):
        for _ in range(n):
            clf.classify(fork_cache(b3.state_cache), qt, labels)
    ms = timed(run, reps=3)
    print(f"   N={n:3d} sequential -> {ms:7.1f}ms total, {ms/n:5.1f}ms/question")
