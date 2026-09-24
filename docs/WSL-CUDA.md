# Running RYOTIDE on Windows (WSL2 + NVIDIA CUDA)

This runs the PyTorch backend on an NVIDIA GPU inside WSL2, checks it against the
reference results in this repo, and serves it the way JevBench measures entrants.
Written for a 12 GB card; larger cards can skip the memory options.

**Status: measured on CUDA.** Every configuration below was run on an NVIDIA RTX PRO
6000 Blackwell (a 24 GB MIG slice) under Ubuntu 24.04, installed exactly as in
step 3. The PyTorch path matches the reference: Gemma 4 E4B bf16 on CUDA agrees with
PyTorch on Apple on 110/111 hard decisions, Qwen3.5-4B bf16 on 40/40. A 12 GB card was
not available, so "fits 12 GB" below means *measured peak* under 12 GB, not a run on
one — leave headroom for Windows and the CUDA context (~0.5–1 GB).

## 1. What fits in 12 GB — measured

Peak GPU memory over a whole hard-tier run, including loading and the longest prompt
(~3,700 tokens). Accuracy is the 111 public hard decisions (MLX 8-bit reference in
brackets).

| model | options | peak GPU | peak system RAM | hard acc | fits 12 GB |
|---|---|---|---|---|---|
| **Qwen3.5-4B** | **`--quant int8`** | **5.7 GB** | 9.5 GB | 76/111 (72) | **yes — recommended** |
| Qwen3.5-4B | bf16 (no flag) | 9.1 GB | 9.4 GB | 73/111 (72) | yes |
| **Gemma 4 E4B** | **`--quant nf4`** | **10.2 GB** | 16.9 GB | 68/111 (68) | **yes, tight** |
| Gemma 4 E4B | `--quant int8` | 12.3 GB | 16.6 GB | 67/111 (68) | no |
| Gemma 4 E4B | bf16 | ~16 GB | | 70/111 | no |

- **Qwen int8** is the comfortable choice: 95.5% of decisions identical to MLX 8-bit.
- **Gemma nf4** fits, but 4-bit is lossier: the same 68/111, yet 12 of 111 individual
  decisions differ from MLX 8-bit (89% identical). Through the wire it scored 183/231
  against MLX 8-bit's 182, 95% of decisions identical.
- **`--low-vram` does not help on a 12 GB card.** It keeps Gemma's per-layer embedding
  tables (2.8 B parameters) in system RAM and runs their lookup there — running memory
  drops to ~6 GB, decisions stay identical (111/111) — but `transformers` still passes
  those weights through the GPU while *loading*, so the load peak stays ~12 GB, and
  system RAM peaks at ~28 GB. Useful only where loading fits and running memory does
  not; not for this card.

## 2. One-time WSL setup

On **Windows** (PowerShell as administrator):

```powershell
wsl --install -d Ubuntu-24.04        # WSL2 + Ubuntu; reboot if asked
```

- Install the **latest NVIDIA driver for Windows** from nvidia.com. Do **not** install
  an NVIDIA/CUDA driver inside Ubuntu — WSL uses the Windows driver.
- Give WSL enough RAM: `--low-vram` keeps ~6 GB of weights in system RAM, and loading
  briefly needs more. Create `%UserProfile%\.wslconfig`:

  ```ini
  [wsl2]
  memory=20GB
  ```

  then run `wsl --shutdown` and reopen Ubuntu.

In **Ubuntu**:

```bash
nvidia-smi                           # must show the GPU; if not, fix the Windows driver first
sudo apt update && sudo apt install -y git build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
```

Work in the Linux file system (`~/`), **not** under `/mnt/c/...` — it is many times
slower for model loading.

## 3. Install

```bash
cd ~ && git clone https://github.com/csabag/ryotide && cd ryotide
uv sync --extra cuda                 # torch (CUDA wheels), transformers, accelerate,
                                     # bitsandbytes, flash-linear-attention
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and your GPU's name. MLX is not installed on Linux;
nothing here needs it.

Model weights (public, Apache-2.0, no token needed) download on first use into
`~/.cache/huggingface`: ~16 GB for Gemma 4 E4B, ~9 GB for Qwen3.5-4B. The full bf16
files are downloaded even for `int8` — quantization happens while loading. Setting
`HF_TOKEN` is optional and only speeds up downloads.

## 4. Run the hard tier and compare

Every command writes `results/jevbench/<tag>/results.jsonl`.

**Gemma 4 E4B, 4-bit** (the Gemma configuration that fits 12 GB):

```bash
uv run python bench/run_jevbench.py --backend torch --device cuda \
  --model google/gemma-4-E4B-it --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --quant nf4 \
  --tasks vendor/jevbench/datasets/public/hard.jsonl \
  --orders 1 --repeat 2 --prefix 'Answer: **' --marker '{}' --tag cuda-gemma-nf4-hard-mine

uv run python bench/compare_runs.py cuda-gemma-nf4-hard cuda-gemma-nf4-hard-mine   # vs our CUDA nf4 run
uv run python bench/compare_runs.py gemma4-rep2-pin2   cuda-gemma-nf4-hard-mine   # vs MLX 8-bit
```

The echo (question repeated after the state) is on for every question by default.
To check CUDA against our PyTorch-on-Apple run, which predates that default, rerun
bf16 with the old gate — only on a card with ~18 GB free, since bf16 is ~16 GB:

```bash
uv run python bench/run_jevbench.py --backend torch --device cuda \
  --model google/gemma-4-E4B-it --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --echo-min-options 2 --tasks vendor/jevbench/datasets/public/hard.jsonl \
  --orders 1 --repeat 2 --prefix 'Answer: **' --marker '{}' --tag cuda-gemma-bf16-gated-hard-mine
uv run python bench/compare_runs.py torch-mps-gemma4-e4b-bf16-hard cuda-gemma-bf16-gated-hard-mine
```

**Qwen3.5-4B, 8-bit:**

```bash
uv run python bench/run_jevbench.py --backend torch --device cuda \
  --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --quant int8 \
  --tasks vendor/jevbench/datasets/public/hard.jsonl \
  --orders 1 --repeat 2 --prefix 'Answer: **' --marker '{}' --tag cuda-qwen-int8-hard-mine

uv run python bench/compare_runs.py cuda-qwen-int8-hard cuda-qwen-int8-hard-mine   # vs our CUDA int8 run
uv run python bench/compare_runs.py 3.5-4b-rep2         cuda-qwen-int8-hard-mine   # vs MLX 8-bit
```

The first Qwen run compiles the `flash-linear-attention` Triton kernels, so the
first few items are slow. Without those kernels `transformers` falls back to pure
PyTorch — correct, but several times slower.

**What to expect.** bitsandbytes `int8` rounds differently from MLX's 8-bit format,
so do not expect bit-identical results. `compare_runs.py` prints a verdict:

- `MATCH` — identical decisions. Unlikely with `int8`; expected with bf16.
- `CLOSE` — ≥ 90% identical decisions and every marker mass ≥ 0.9. **This is a pass.**
  Accuracy should be within a few items of the reference (hard tier, echo on every
  question: Gemma 68/111 and Qwen 72/111 on MLX 8-bit; the gated bf16 check: Gemma
  70/111 on PyTorch/Apple).
- `INVESTIGATE` — below 90% agreement or a marker mass under 0.9: the read position
  or the weights are not what we think. Send us the output.

## 5. The wire format (how JevBench measures)

JevBench drives entrants over HTTP with its stock `typesafe` adapter. To reproduce
that, start the server in one terminal:

```bash
PYTHONPATH=src uv run python -m ryotide.server --backend torch --device cuda \
  --model google/gemma-4-E4B-it --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --quant nf4
# wait for: ryotide ready ...
curl -s localhost:8778/health             # engine, revision, prompt hash, quant, offloaded modules
```

and run all 231 public decisions from a second terminal:

```bash
cd vendor/jevbench && mkdir -p ../../results/jevbench/cuda-wire-gemma-nf4-mine
PYTHONPATH=. uv run python -m jevbench.cli run --adapter typesafe \
  --endpoint http://127.0.0.1:8778 --key-env '' --model ryotide --reserve-usd 0 \
  --tasks datasets/public/original.jsonl,datasets/public/easy.jsonl,datasets/public/hard.jsonl \
  --results ../../results/jevbench/cuda-wire-gemma-nf4-mine/results.jsonl \
  --ledger ../../results/jevbench/cuda-wire-gemma-nf4-mine/ledger.jsonl --raw-dir /tmp/raw
cd ../.. && uv run python bench/compare_runs.py cuda-wire-gemma-nf4 cuda-wire-gemma-nf4-mine
```

Reference: 183/231 through the wire on CUDA nf4 (`cuda-wire-gemma-nf4`), 182/231 on
MLX 8-bit, both with the default settings (echo on every question, natural option
order).

## 6. What to send back

- `results/jevbench/cuda-*/results.jsonl` and `summary.json` (the runner records
  peak GPU memory there as `cuda_peak_gb`, and prints it at the end)
- the full output of each `compare_runs.py`
- `curl -s localhost:8778/health` from the server run
- `nvidia-smi` and
  `uv run python -c "import torch, transformers, bitsandbytes; print(torch.__version__, transformers.__version__, bitsandbytes.__version__)"`

A pull request with the `results/jevbench/cuda-*` directories is the easiest way.

## 7. Troubleshooting

| symptom | fix |
|---|---|
| `torch.cuda.is_available()` is `False` | `nvidia-smi` inside WSL must work first; update the Windows driver; never install a Linux NVIDIA driver in WSL |
| `CUDA out of memory` while loading | Gemma: `--quant nf4` (int8 needs ~12.3 GB); Qwen: `--quant int8`; close other GPU programs (`--low-vram` does not lower the load peak) |
| process killed while loading (no error) | WSL ran out of system RAM: raise `memory=` in `.wslconfig`, then `wsl --shutdown` |
| bitsandbytes: "CUDA setup failed" / cannot find `libcuda.so` | `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` |
| Qwen very slow | `flash-linear-attention` missing or failed to build: `uv sync --extra cuda` again and read its output |
| `--quant ... needs CUDA` | you passed `--quant` without `--device cuda`, or CUDA is not visible |
