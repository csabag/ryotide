"""JevBench adapter for the MLX single-forward-pass classifier.

JevBench (https://github.com/fstandhartinger/jevbench, MIT) is Benchmark Heaven's
benchmark for Jev-class typed decision models: a state plus a bounded rubric in,
a probability per exact label out. It is the benchmark this project should be
measured on -- GLUE was a proxy for a claim nobody actually made.

Mapping a JevBench record onto the Jev-style primitive:

    state                    -> KV-cache prefill (shared across option orders)
    instructions + criteria  -> option block, one lettered marker per label
    labels                   -> letter markers A, B, C ... (single tokens)
    answer                   -> softmax over those letters at ONE position

Labels themselves are usually multi-token ("change_address"), so they cannot be
scored directly at a single position. Lettered option markers are the standard
fix and what the other local entrants do -- the distribution is the model's own
softmax over the markers, never verbalized, never logprob-parsed.

ORDER DEBIASING. JevBench's own README records open-alternative-jev scoring 72%
with `A. yes, B. no` and 21% with the order reversed. Option-order bias is the
dominant failure mode at this model size, so by default we evaluate each decision
under several label orders and average the per-label probabilities. This is nearly
free here and nowhere else: every order shares the same state prefix, so the state
is prefilled once and each order is a branch off that cache. The state is the
expensive part; the option block is a few dozen tokens.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from .classify import Classifier

# Backends are imported lazily: MLX only exists on Apple Silicon, and a CUDA host
# must be able to import this module (and serve decisions) without it.

LETTERS = "ABCDEFGHIJKLMNOP"


def _state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True)


def _criterion_for(task_question: dict, label: str, index: int) -> str | None:
    """Pull the rubric line for one label, across all three question types."""
    crit = task_question.get("criteria")
    if crit is None:
        return None
    if isinstance(crit, list):              # score: one line per level, in order
        return crit[index] if index < len(crit) else None
    if isinstance(crit, dict):
        if label in crit:                   # choice: keyed by label
            return crit[label]
        # noul: keyed "true"/"false" rather than "yes"/"no"
        return crit.get({"yes": "true", "no": "false"}.get(label, label))
    return None


IDK_TEXT = "cannot be determined from the information given"
DEFAULT_INSTRUCTION = "Reply with the letter of the single best option."
_ABSTAINISH = ("cannot", "unknown", "insufficient", "undetermined", "unclear")


def _has_abstain(task) -> bool:
    return any(any(p in l.lower() for p in _ABSTAINISH) for l in task.labels)


def _orders(n: int, k: int) -> list[list[int]]:
    """Label orders to average over. Identity, reversed, then rotations."""
    if k <= 1:
        return [list(range(n))]
    out = [list(range(n)), list(range(n - 1, -1, -1))]
    r = 1
    while len(out) < k and r < n:
        rot = [(i + r) % n for i in range(n)]
        if rot not in out:
            out.append(rot)
        r += 1
    return out[:k]


class MlxJevLocalAdapter:
    """In-process MLX entrant. Native softmax over option markers, no network."""

    name = "mlx_jev_local"
    cost_basis = "local_gpu_no_provider_tariff"

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        key_env: str = "",
        timeout_s: float | None = None,
        price_input_per_m: float | None = None,
        price_output_per_m: float | None = None,
        revision: str | None = None,
        orders: int = 2,
        chat: bool = True,
        franken: tuple | None = None,   # (delta, shift) -> duplicate query heads
        dtype: str | None = None,
        idk: bool = False,              # add an unscored "cannot determine" option
        idk_first: bool = False,        # place it at slot A instead of last
        repeat: int = 1,                # emit the question body N times
        pin_prefix: str | None = None,  # skip calibration, force this read position
        pin_marker: str | None = None,
        instruction: str | None = None,  # override the closing instruction line
        question_first: bool = False,    # also emit the question BEFORE the state
        markers: Sequence[str] | None = None,  # option markers; default A, B, C, ...
        backend: str = "mlx",            # "mlx" (reference) or "torch" (CUDA / MPS / CPU)
        device: str | None = None,       # torch only; default: cuda > mps > cpu
        echo_min_options: int = 2,       # echo only when there are MORE options than this
    ):
        # `endpoint` carries the model id/path, matching the other local adapters.
        self.path = endpoint or model or "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        self.model = model or self.path
        self.key_env = key_env
        # Local weights have no provider tariff. Null, never 0 -- unmetered is not free.
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.revision = revision
        self.orders = max(1, int(orders))
        self.chat = chat
        self.franken = franken
        self.dtype = dtype
        self.idk = idk
        self.idk_first = idk_first
        self.repeat = max(1, int(repeat))
        self.repeat_min_options = echo_min_options   # repeat only if MORE than this many options
        self.instruction = instruction or DEFAULT_INSTRUCTION
        self.question_first = question_first
        self.markers = list(markers) if markers else list(LETTERS)
        if backend not in ("mlx", "torch"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.device = device
        self._clf: Classifier | None = None
        self._letter_ids: dict[int, list[int]] = {}
        self._suffix: str | None = None      # chosen by _calibrate_read_position
        self._pattern: str = " {}"           # marker surface form, chosen with it
        self._suffix_mass: float = 0.0
        self._calib_pool: list = []
        if pin_prefix is not None:
            # Pinned so an A/B changes ONE thing: letting the sweep re-choose
            # would vary the read position alongside whatever is under test.
            # Must come AFTER the defaults above, or it gets overwritten.
            self._suffix = pin_prefix
            self._pattern = pin_marker or "{}"

    # -- setup (called before the clock via JEVBENCH_WARM_LOAD=1) -----------

    def load(self) -> Classifier:
        if self._clf is None and self.backend == "torch":
            from .torch_backend import TorchClassifier
            self._clf = TorchClassifier(self.path, revision=self.revision,
                                        device=self.device, dtype=self.dtype or "bfloat16")
        if self._clf is None:
            import mlx.core as mx
            from .classify import Classifier
            from .model import load_model
            model, tok = load_model(self.path)
            if self.dtype:
                model.set_dtype(getattr(mx, self.dtype))
            if self.franken:
                from .franken import frankenize
                frankenize(model, delta=self.franken[0], shift=self.franken[1])
            self._clf = Classifier(model, tok)
        return self._clf

    # Candidate answer prefixes. Models differ in where the answer token lands:
    # Qwen2.5 emits " A" straight after "Answer:", Qwen3 wants to open markdown
    # bold first ("**A**"), reasoning models need their think block closed. Rather
    # than special-case each family, probe them and keep whichever puts the most
    # probability mass on the option markers.
    # Answer prefixes to sweep. Families differ in where the answer token lands:
    # Qwen2.5 emits " A" after "Answer:", Qwen3/Gemma open markdown bold first,
    # and harmony-format models (gpt-oss) must have their output channel opened
    # or the next token is "<|channel|>" rather than any option marker.
    SUFFIXES = ("Answer:", "Answer: **", "**", "Answer:\n", "The answer is",
                "Answer: (", "",
                "<|channel|>final<|message|>",
                "<|channel|>final<|message|>Answer: **")

    PATTERNS = (" {}", "{}")

    CALIB_TASKS = 6

    def calibrate(self, tasks) -> str:
        """Calibrate up front, before any decision is timed. Pass a handful of
        representative tasks; every decision then uses the same read position."""
        self._calib_pool = list(tasks)[: self.CALIB_TASKS]
        return self._calibrate_read_position()

    def _calibrate_read_position(self, task=None) -> str:
        """Choose the (answer prefix, marker surface form) pair that lands on an
        option marker for THIS model, averaged over several real tasks.

        The two are coupled -- after "Answer:" the next token is " A", after
        "Answer: **" it is "A" -- so they must be swept together. And a single
        task is not enough: the same prefix measured 0.9988 on one item and
        0.0001 across the set, so the choice has to be averaged over a handful
        of real prompts (and over differing option counts) before it is trusted.
        """
        if self._suffix is not None:
            return self._suffix
        pool = (self._calib_pool or ([task] if task else []))[: self.CALIB_TASKS]
        if not pool:
            self._suffix, self._pattern = "Answer:", " {}"
            return self._suffix
        best = (self.SUFFIXES[0], self.PATTERNS[0], -1.0)
        for pat in self.PATTERNS:
            for suf in self.SUFFIXES:
                masses = []
                for t in pool:
                    n = len(t.labels)
                    try:
                        toks = self._tokens(self._prompt(t, list(range(n))), suffix=suf)
                        masses.append(self._read(toks, self._letters(n, pat))[1])
                    except Exception:
                        masses.append(0.0)
                m = sum(masses) / len(masses)
                # Prefer the SHORTEST prefix among those statistically tied with
                # the best: a longer one is prefilled on every decision for no
                # gain. Qwen3.5-9B otherwise picks a 9-token harmony prefix when
                # a 3-token one reads identically.
                if m > best[2] + 0.01 or (m > best[2] - 0.01 and len(suf) < len(best[0])):
                    if m > best[2] - 0.01:
                        best = (suf, pat, max(m, best[2]))
        self._suffix, self._pattern, self._suffix_mass = best
        print(f"[calibrated] prefix={self._suffix!r} marker={self._pattern!r} "
              f"mean marker mass {self._suffix_mass:.4f} over {len(pool)} tasks", flush=True)
        return self._suffix

    def _read(self, toks: Sequence[int], ids: Sequence[int], cache=None) -> tuple[list[float], float]:
        """Masked next-token distribution at the end of `toks`, plus marker mass.

        The one place the backends differ. MLX may extend a prefilled `cache`
        (a branch off a shared prefix); torch always runs one full forward pass.
        """
        clf = self._clf
        if self.backend == "torch":
            return clf.read(toks, ids)
        import mlx.core as mx
        row = clf.logits_at_end(cache if cache is not None else clf.prefill([]), toks)
        sel = mx.array(list(ids))
        mass = float(mx.sum(mx.softmax(row.astype(mx.float32))[sel]).item())
        return mx.softmax(row[sel].astype(mx.float32)).tolist(), mass

    def _letters(self, n: int, pattern: str | None = None) -> list[int]:
        """Token ids for the first n option markers under one surface form."""
        pat = pattern or self._pattern
        key = (n, pat, tuple(self.markers[:n]))
        if key in self._letter_ids:
            return self._letter_ids[key]
        tok = self._clf.tokenizer
        ids = [tok.encode(pat.format(self.markers[i]), add_special_tokens=False)
               for i in range(n)]
        if all(len(x) == 1 for x in ids) and len({x[0] for x in ids}) == n:
            self._letter_ids[key] = [x[0] for x in ids]
            return self._letter_ids[key]
        raise ValueError(f"no single-token marker set for {n} options under {pat!r}")

    # -- prompt ------------------------------------------------------------

    def _prompt(self, task, order: Sequence[int]) -> str:
        q = task.question
        idk = self._idk_active(task)
        # Slot 0 when idk_first, else the final slot. Position is load-bearing:
        # correct answers sit at index 0 on 27.9% of hard items and index 4 on
        # 1.8%, so an escape hatch parked last is one the model rarely considers.
        offset = 1 if (idk and self.idk_first) else 0
        lines = []
        if idk and self.idk_first:
            lines.append(f"{self.markers[0]}. {IDK_TEXT}")
        for slot, li in enumerate(order):
            label = task.labels[li]
            crit = _criterion_for(q, label, li)
            lines.append(f"{self.markers[slot + offset]}. {label}" + (f" - {crit}" if crit else ""))
        if idk and not self.idk_first:
            lines.append(f"{self.markers[len(order)]}. {IDK_TEXT}")
        body = (
            f"{q['instructions']}\n\n"
            f"Options:\n" + "\n".join(lines) + "\n\n"
            + self.instruction
        )
        # Repeating the question body makes the second copy effectively
        # bidirectional over it: under causal attention the first copy builds
        # each option's representation without having seen the later options,
        # while every token of the second copy attends to all of them. A typed
        # decision is comparative, so the options should be able to see each
        # other. The state is never repeated -- it is the expensive part.
        # Only when there is something to compare. With two options the second
        # copy reveals nothing the first did not already contain -- it is pure
        # duplication between the state and the read position, and measurably
        # hurts some models on binary items while helping every model on
        # multi-option ones.
        if self.repeat > 1 and len(task.labels) > self.repeat_min_options:
            body = ("\n\n".join([body] * self.repeat))
        state = _state_text(task.state)
        # Under causal attention the state is encoded having never seen the
        # question: every state token attends only backwards. A leading copy of
        # the question makes the state's own representation question-aware,
        # which is a different mechanism from the trailing echo (that one makes
        # the OPTIONS mutually visible). The state is still emitted once -- it
        # is the expensive part -- so Q/state/Q costs the same as state/Q/Q.
        if self.question_first:
            return body + "\n\n" + state + "\n\n" + body
        return state + "\n\n" + body

    def _idk_active(self, task) -> bool:
        """Add the escape hatch only where the task does not already have one."""
        return self.idk and not _has_abstain(task)

    def _tokens(self, text: str, suffix: str | None = None) -> list[int]:
        """Prompt tokens ending exactly where the decision is read.

        Reasoning models (Qwen3.5) open a <think> block in their generation
        prompt, which would put our read position INSIDE the reasoning rather
        than at the answer: the model then puts its mass on how to start
        thinking, and only ~15% reaches the option markers. Asking the template
        to skip thinking restores ~98%. Templates that don't take the argument
        raise, so fall back rather than assume.
        """
        tok = self._clf.tokenizer
        suf = suffix if suffix is not None else (self._suffix or "Answer:")
        if not self.chat:
            return tok.encode(text + "\n" + suf, add_special_tokens=False)
        msg = [{"role": "user", "content": text}]
        try:
            ids = tok.apply_chat_template(msg, add_generation_prompt=True,
                                          enable_thinking=False)
        except (TypeError, ValueError):
            ids = tok.apply_chat_template(msg, add_generation_prompt=True)
        if hasattr(ids, "keys"):          # transformers >= 5 returns a mapping
            ids = ids["input_ids"]
        return list(ids) + (tok.encode(suf, add_special_tokens=False) if suf else [])

    def build_request(self, task) -> dict:
        return {
            "state": _state_text(task.state),
            "questions": {"decision": {k: task.question[k] for k in
                                       ("type", "instructions") if k in task.question}},
            "orders": self.orders,
            "chat_template": self.chat,
        }

    # -- the decision ------------------------------------------------------

    def run(self, task):
        from jevbench.adapters.base import DecisionResult  # harness type

        res = DecisionResult(adapter=self.name, ok=False,
                             probs_source="native", model=self.model)
        res.request_body = self.build_request(task)
        try:
            clf = self.load()
        except Exception as e:
            res.error = f"load failed: {type(e).__name__}: {str(e)[:250]}"
            return res

        n = len(task.labels)
        t0 = time.perf_counter()
        try:
            self._calibrate_read_position(task)
            letters = self._letters(n)
            orders = _orders(n, self.orders)
            prompts = [self._tokens(self._prompt(task, o)) for o in orders]

            # Every order shares the state prefix: prefill it once, branch per order.
            # (MLX only -- the torch backend runs each order as one full pass.)
            shared = 0
            if len(prompts) > 1 and self.backend == "mlx":
                shortest = min(len(p) for p in prompts)
                while shared < shortest and len({p[shared] for p in prompts}) == 1:
                    shared += 1
                shared = max(0, shared - 1)   # keep one token for the branch to extend

            base = clf.prefill(prompts[0][:shared]) if shared else None
            if shared:
                from .branch import fork_cache

            totals = [0.0] * n
            marker_mass: list[float] = []
            idk_probs: list[float] = []
            use_idk = self._idk_active(task)
            read = self._letters(n + 1) if use_idk else letters
            for order, toks in zip(orders, prompts):
                branch = fork_cache(base) if shared else None
                p, mass = self._read(toks[shared:], read, cache=branch)
                marker_mass.append(mass)
                if use_idk:
                    if self.idk_first:
                        idk_probs.append(float(p[0]))
                        keep = sum(p[1:]) or 1.0
                        p = [x / keep for x in p[1:]]
                    else:
                        idk_probs.append(float(p[-1]))
                        keep = sum(p[:-1]) or 1.0
                        p = [x / keep for x in p[:-1]]
                for slot, li in enumerate(order):
                    totals[li] += float(p[slot])   # fold back onto the true label
                del branch

            total = sum(totals) or 1.0
            probs = {task.labels[i]: totals[i] / total for i in range(n)}
        except Exception as e:
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res

        res.latency_s = time.perf_counter() - t0
        res.probs = probs
        res.label = max(probs, key=probs.get)
        res.ok = True
        res.usage = {"input_tokens": int(sum(len(p) for p in prompts)),
                     "output_tokens": 0}
        res.raw = {
            "response": {"probs": probs},
            "runtime": {
                "device": "mlx-metal" if self.backend == "mlx" else f"torch-{clf.device}",
                "model": self.model,
                "orders_averaged": len(orders),
                "franken": list(self.franken) if self.franken else None,
                "dtype": self.dtype,
                "shared_prefix_tokens": shared,
                "probability_origin": "native-softmax-over-option-markers",
                "answer_prefix": self._suffix,
                "marker_pattern": self._pattern,
                "idk_prob": (round(sum(idk_probs) / len(idk_probs), 5)
                             if idk_probs else None),
                "idk_position": ("first" if self.idk_first else "last") if use_idk else None,
                "question_repeats": (self.repeat if len(task.labels) > self.repeat_min_options else 1),
                "question_first": self.question_first,
                "instruction": self.instruction,
                # Share of the FULL vocab distribution sitting on the markers.
                # Near 1.0 means the prompt lands the read at a real answer
                # position; a low value means it does not, and the decision is
                # being renormalised out of noise.
                "marker_mass": round(sum(marker_mass) / len(marker_mass), 4),
            },
        }
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0
