"""JevBench v1.1.3 charts (v1.1 charts + the GPU round), drawn only from jevbench-v1.1.3-results.json.

    python3 v113/render_charts_v113.py

main-score.png     the composite, ranked, with each axis' contribution beside it
sub-benchmarks.png Capability / Speed / Cost as three small multiples (one axis each)
tiers.png          accuracy per tier (easy / standard / judge)
sensitivity.png    rank of every system under six weightings

Colour carries one thing: what kind of system a row is. Four categorical slots,
validated all-pairs on the light surface (worst CVD dE 9.2, worst normal 16.3).
Every bar is also labelled with its name, so identity never rests on colour.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

JOB = Path(__file__).resolve().parents[1]
RES = json.loads((JOB / "v113/out/jevbench-v1.1.3-results.json").read_text())
OUT = JOB / "v113/out/charts"
OUT.mkdir(parents=True, exist_ok=True)

CLASS = {
    "jev": ("#2a78d6", "Jev (TypeSafe, closed)"),
    "jev-rebuild": ("#eb6834", "Open Jev rebuild"),
    "llm-baseline": ("#1baf7a", "Instruction model, JSON schema"),
    "small-tool-model": ("#4a3aa7", "Small tool-calling model"),
}
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2", "#b9b8b1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.parse_math": False, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

SHORT = {
    "jev-1.13.0": "Jev 1.13.0", "openjev-sglang": "openjev-sglang", "system-one-open": "system-one-open",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "gpt-5.6-luna": "GPT-5.6 Luna (low)",
    "deepseek-flash": "DeepSeek V4.1 Flash", "open-jev-deberta-v3-large": "open-jev-deberta-v3-large",
    "needle-3": "Needle 3", "needle-3-tools": "Needle 3, options as tools*",
    "qwen3.8-27b": "Qwen3.8 27B", "open-alternative-jev": "open-alternative-jev (GPU)",
    "openjev-razorback16": "OpenJev, DiffusionGemma (GPU)", "semif-qwen3.5-4b": "SemIf, Qwen3.5-4B (GPU)",
    "nimble-9b": "Bespoke Nimble 9B (GPU)", "system-one-sg": "system-one, Qwen3-8B (GPU)", "open-alternative-jev-yesfirst": "open-alternative-jev, yes-first* (GPU)",
}


def name(s):
    n = SHORT.get(s["key"], s["display"])
    return n + (" (partial run)" if s["partial"] else "")


def systems():
    """Ranked systems first by Main Score, then partial runs that still have a score."""
    rk = [s for s in RES["systems"] if s["ranked"]]
    part = [s for s in RES["systems"] if not s["ranked"] and s["main_score"] is not None]
    return rk, part


def frame(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def legend(fig, used, loc="lower center", ncol=4, extra=()):
    h = [Patch(facecolor=CLASS[k][0], label=CLASS[k][1]) for k in CLASS if k in used]
    fig.legend(handles=h + list(extra), loc="lower left", bbox_to_anchor=(0.01, 0.055), ncol=ncol, frameon=False,
               fontsize=10, labelcolor=INK2)


def footer(fig, text):
    fig.text(0.01, 0.005, text, fontsize=8.5, color=INK2, ha="left", va="bottom")


def main_chart():
    rk, part = systems()
    rows = rk + part
    fig, ax = plt.subplots(figsize=(12, 0.52 * len(rows) + 2.4), facecolor=SURFACE)
    frame(ax)
    y = list(range(len(rows)))[::-1]
    for yy, s in zip(y, rows):
        col = CLASS[s["class"]][0]
        ax.barh(yy, s["main_score"], height=0.62, color=col if s["ranked"] else "none",
                edgecolor=col, hatch=None if s["ranked"] else "////", linewidth=1.2 if not s["ranked"] else 0)
        c, sp, k = s["capability"]["score"], s["speed"]["score"], s["cost"]["score"]
        ax.text(s["main_score"] + 1, yy, f"{s['main_score']:.1f}", va="center", fontsize=11, color=INK, fontweight="bold")
        ax.text(101.5, yy, f"{c:5.1f}    {sp:5.1f}    {k:5.1f}" + ("   est." if s["cost"]["kind"] == "estimate" else ""),
                va="center", fontsize=10, color=INK2, family="DejaVu Sans Mono")
    ax.text(101.5, len(rows) - 0.35, "Capab.   Speed    Cost", fontsize=9.5, color=INK2, family="DejaVu Sans Mono")
    ax.set_yticks(y, [name(s) for s in rows], fontsize=11, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("JevBench Main Composite Score = (Capability + Speed + Cost) / 3  (Balanced 33:33:33, each 0-100)")
    ax.set_title("JevBench v1.1.3 - Main Composite Score (Balanced 33:33:33)", loc="left", fontsize=16, color=INK, pad=14, fontweight="bold")
    fig.subplots_adjust(left=0.27, right=0.74, top=0.93, bottom=0.2)
    legend(fig, {s["class"] for s in rows}, ncol=2, loc="lower left",
           extra=[Patch(facecolor="none", edgecolor=INK2, hatch="////", label="Partial run - shown, not ranked")])
    footer(fig, "314 typed decisions per system (72 easy / 96 standard / 146 judge). est. = no tariff for us: priced like a large inference provider hosting a model of that size (OpenRouter/DeepInfra list prices).\n"
                "(GPU) = v1.1.3 GPU round: self-hosted on a rented RunPod GPU, Speed measured from Germany over the internet to it. * = adapter mode added after the frozen run,\n"
                "reported beside it (Needle 3 options-as-tools; open-alternative-jev with the author's yes/no order). github.com/fstandhartinger/jevbench")
    fig.savefig(OUT / "main-score.png", dpi=160)
    plt.close(fig)


def sub_chart():
    rk, part = systems()
    rows = rk + part
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.5 * len(rows) + 2.6), facecolor=SURFACE, sharey=True)
    y = list(range(len(rows)))[::-1]
    specs = [("capability", "Capability", lambda s: f"{s['capability']['score']:.1f}"),
             ("speed", "Speed", lambda s: f"{s['speed']['p50_s']:.2f} s / {s['speed']['p95_s']:.2f} s"),
             ("cost", "Cost", lambda s: ("~" if s["cost"]["kind"] == "estimate" else "") + f"${s['cost']['usd_per_1000']:.3f}")]
    for ax, (key, title, lab) in zip(axes, specs):
        frame(ax)
        for yy, s in zip(y, rows):
            col = CLASS[s["class"]][0]
            v = s[key]["score"]
            est = key == "cost" and s["cost"]["kind"] == "estimate"
            ax.barh(yy, v, height=0.62, color="none" if (est or not s["ranked"]) else col, edgecolor=col,
                    hatch=("...." if est else None) if s["ranked"] else "////", linewidth=1.2)
            ax.text(min(v, 100) + 1.5, yy, lab(s), va="center", fontsize=9.5, color=INK)
        ax.set_xlim(0, 135)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(title + " score (0-100)", loc="left", fontsize=13, color=INK)
    axes[0].set_yticks(y, [name(s) for s in rows], fontsize=10.5, color=INK)
    axes[1].set_xlabel("label: median / p95 latency, serial, network incl.")
    axes[2].set_xlabel("label: $ per 1,000 decisions (~ = estimate)")
    axes[0].set_xlabel("mean of easy, standard, judge tier accuracy")
    fig.suptitle("JevBench v1.1.3 - the three sub-benchmarks", x=0.01, ha="left", fontsize=16, color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.17, right=0.99, top=0.88, bottom=0.25, wspace=0.08)
    legend(fig, {s["class"] for s in rows}, ncol=5,
           extra=[Patch(facecolor="none", edgecolor=INK2, hatch="....", label="Cost estimated (stated reference)")])
    footer(fig, "Speed: 0.1 s = 100, 1 s = 50, 10 s = 0 (log), mean of p50 and p95 scores. Cost: $0.01 = 100, $0.10 = 67, $1 = 33, $10 = 0 per 1,000 decisions (log).")
    fig.savefig(OUT / "sub-benchmarks.png", dpi=160)
    plt.close(fig)


def tier_chart():
    rk, part = systems()
    rows = rk + part
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.5 * len(rows) + 2.4), facecolor=SURFACE, sharey=True)
    y = list(range(len(rows)))[::-1]
    for ax, t in zip(axes, ("easy", "standard", "judge")):
        frame(ax)
        for yy, s in zip(y, rows):
            col = CLASS[s["class"]][0]
            a = s["capability"]["tier_accuracy"][t]
            if a is None:
                continue
            ax.barh(yy, 100 * a, height=0.62, color=col if s["ranked"] else "none", edgecolor=col,
                    hatch=None if s["ranked"] else "////", linewidth=1.2)
            ax.text(100 * a + 1.5, yy, f"{100 * a:.1f}%", va="center", fontsize=9.5, color=INK)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(f"{t.capitalize()} tier ({RES['tiers'][t]} decisions)", loc="left", fontsize=13, color=INK)
    axes[0].set_yticks(y, [name(s) for s in rows], fontsize=10.5, color=INK)
    fig.suptitle("JevBench v1.1.3 - accuracy by tier", x=0.01, ha="left", fontsize=16, color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.18, right=0.99, top=0.88, bottom=0.2, wspace=0.08)
    legend(fig, {s["class"] for s in rows}, ncol=4)
    footer(fig, "Easy: clear-cut intent, explicit yes/no facts, enum extraction, one-obvious-tool selection. Standard: v1.0's authored decisions. Judge: routing real prompts, judging saved math answers.")
    fig.savefig(OUT / "tiers.png", dpi=160)
    plt.close(fig)


def sensitivity_chart():
    rk, _ = systems()
    names = RES["sensitivity_order"]
    fig, ax = plt.subplots(figsize=(13, 0.5 * len(rk) + 2.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_xlim(-0.5, len(names) + 2.2)
    ax.set_ylim(-0.8, len(rk) + 0.6)
    for j, n in enumerate(names):
        ax.text(j + 2.2, len(rk) + 0.1, n.replace(" (headline)", "\n(headline)").replace(" geometric", "\ngeometric").replace(" emphasis", "\nemphasis"),
                ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold" if j == 0 else "normal")
    for i, s in enumerate(rk):
        yy = len(rk) - 1 - i
        ax.add_patch(plt.Rectangle((-0.45, yy - 0.3), 0.18, 0.6, color=CLASS[s["class"]][0]))
        ax.text(-0.2, yy, name(s), va="center", fontsize=11, color=INK)
        for j, n in enumerate(names):
            r = s["rank_under"][n]
            ax.text(j + 2.2, yy, f"#{r}  {s['sensitivity'][n]:.0f}", ha="center", va="center", fontsize=11,
                    color=INK if r == s["rank_under"][names[0]] else "#b3261e",
                    fontweight="bold" if j == 0 else "normal")
    ax.set_title("JevBench v1.1.3 - how the ranking moves with other weights (rank, score)", loc="left",
                 fontsize=15, color=INK, fontweight="bold")
    fig.text(0.01, 0.01, "Weights are Capability / Speed / Cost. Red = rank differs from the headline weighting. Ranked systems only.",
             fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.06)
    fig.savefig(OUT / "sensitivity.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main_chart()
    sub_chart()
    tier_chart()
    sensitivity_chart()
    print("charts ->", OUT)
