# Running RYOTIDE on Windows (WSL2 + NVIDIA CUDA)

This runs the PyTorch backend on an NVIDIA GPU inside WSL2, checks it against the
reference results in this repo, and serves it the way JevBench measures entrants.
Written for a 12 GB card; larger cards can skip the memory options.

**Status:** the PyTorch path matches MLX on Apple Silicon (Gemma 4 E4B 111/111 hard
decisions, Qwen3.5-4B 40/40). The CUDA-specific parts — bitsandbytes `int8` / `nf4`
and the `flash-linear-attention` kernels for Qwen — have **not been run yet**. This
guide is how we find out.

## 1. What fits in 12 GB

| model | options | weights on GPU | fits 12 GB |
|---|---|---|---|
| Gemma 4 E4B | bf16 | ~16 GB | no |
| Gemma 4 E4B | `--quant int8` | ~10.8 GB | too tight |
| **Gemma 4 E4B** | **`--quant int8 --low-vram`** | **~5–6 GB** | **yes (recommended)** |
| Gemma 4 E4B | `--quant nf4 --low-vram` | ~3–4 GB | yes (4-bit, less faithful) |
| Qwen3.5-4B | bf16 | ~8–9 GB | yes, if nothing else uses the GPU |
| **Qwen3.5-4B** | **`--quant int8`** | **~5 GB** | **yes (recommended)** |

Why Gemma needs `--low-vram`: 2.8 B of its 7.9 B parameters are per-layer embedding
tables, and bitsandbytes quantizes Linear layers only, so they would stay in bf16 on
the GPU. `--low-vram` keeps those tables — a lookup of a few rows per token — plus
the unused audio and vision towers in system RAM. On Apple MPS this offload was
checked to change nothing: 20/20 identical decisions, zero probability difference.

Add ~1 GB for the CUDA context and activations. Close other GPU-heavy programs.

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

**Gemma 4 E4B, 8-bit, low VRAM** (the configuration we would submit):

```bash
uv run python bench/run_jevbench.py --backend torch --device cuda \
  --model google/gemma-4-E4B-it --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --quant int8 --low-vram \
  --tasks vendor/jevbench/datasets/public/hard.jsonl \
  --orders 1 --repeat 2 --prefix 'Answer: **' --marker '{}' --tag cuda-gemma-int8-hard

uv run python bench/compare_runs.py gemma4-repcond        cuda-gemma-int8-hard   # vs MLX 8-bit
uv run python bench/compare_runs.py torch-mps-gemma4-e4b-bf16-hard cuda-gemma-int8-hard   # vs torch bf16
```

**Qwen3.5-4B, 8-bit:**

```bash
uv run python bench/run_jevbench.py --backend torch --device cuda \
  --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --quant int8 \
  --tasks vendor/jevbench/datasets/public/hard.jsonl \
  --orders 1 --repeat 2 --prefix 'Answer: **' --marker '{}' --tag cuda-qwen-int8-hard

uv run python bench/compare_runs.py 3.5-4b-repcond cuda-qwen-int8-hard               # vs MLX 8-bit
```

The first Qwen run compiles the `flash-linear-attention` Triton kernels, so the
first few items are slow. Without those kernels `transformers` falls back to pure
PyTorch — correct, but several times slower.

**What to expect.** bitsandbytes `int8` rounds differently from MLX's 8-bit format,
so do not expect bit-identical results. `compare_runs.py` prints a verdict:

- `MATCH` — identical decisions. Unlikely with `int8`; expected with bf16.
- `CLOSE` — ≥ 90% identical decisions and every marker mass ≥ 0.9. **This is a pass.**
  Accuracy should be within a few items of the reference (Gemma hard 71/111 on MLX
  8-bit, 70/111 in bf16; Qwen hard 71/111 on MLX 8-bit).
- `INVESTIGATE` — below 90% agreement or a marker mass under 0.9: the read position
  or the weights are not what we think. Send us the output.

## 5. The wire format (how JevBench measures)

JevBench drives entrants over HTTP with its stock `typesafe` adapter. To reproduce
that, start the server in one terminal:

```bash
PYTHONPATH=src uv run python -m ryotide.server --backend torch --device cuda \
  --model google/gemma-4-E4B-it --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --quant int8 --low-vram
# wait for: ryotide ready ...
curl -s localhost:8778/health             # engine, revision, prompt hash, quant, offloaded modules
```

and run all 231 public decisions from a second terminal:

```bash
cd vendor/jevbench && mkdir -p ../../results/jevbench/cuda-wire-gemma-int8
PYTHONPATH=. uv run python -m jevbench.cli run --adapter typesafe \
  --endpoint http://127.0.0.1:8778 --key-env '' --model ryotide --reserve-usd 0 \
  --tasks datasets/public/original.jsonl,datasets/public/easy.jsonl,datasets/public/hard.jsonl \
  --results ../../results/jevbench/cuda-wire-gemma-int8/results.jsonl \
  --ledger ../../results/jevbench/cuda-wire-gemma-int8/ledger.jsonl --raw-dir /tmp/raw
cd ../.. && uv run python bench/compare_runs.py wire-gemma4-e4b-8bit-natural cuda-wire-gemma-int8
```

Reference: 185/231 through the wire on MLX 8-bit.

## 6. What to send back

- `results/jevbench/cuda-*/results.jsonl` (and `summary.json` where written)
- the full output of each `compare_runs.py`
- `curl -s localhost:8778/health` from the server run
- `nvidia-smi` and
  `uv run python -c "import torch, transformers, bitsandbytes; print(torch.__version__, transformers.__version__, bitsandbytes.__version__)"`

A pull request with the `results/jevbench/cuda-*` directories is the easiest way.

## 7. Troubleshooting

| symptom | fix |
|---|---|
| `torch.cuda.is_available()` is `False` | `nvidia-smi` inside WSL must work first; update the Windows driver; never install a Linux NVIDIA driver in WSL |
| `CUDA out of memory` while loading | add `--low-vram`; or use `--quant nf4`; close other GPU programs |
| process killed while loading (no error) | WSL ran out of system RAM: raise `memory=` in `.wslconfig`, then `wsl --shutdown` |
| bitsandbytes: "CUDA setup failed" / cannot find `libcuda.so` | `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` |
| Qwen very slow | `flash-linear-attention` missing or failed to build: `uv sync --extra cuda` again and read its output |
| `--quant ... needs CUDA` | you passed `--quant` without `--device cuda`, or CUDA is not visible |
