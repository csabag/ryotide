# RYOTIDE decision server, PyTorch backend on NVIDIA GPUs.
#
#   docker build -t ryotide .
#   docker run --gpus all -p 127.0.0.1:8778:8778 \
#     -v ryotide-hf:/root/.cache/huggingface ryotide --preset ryotide-gemma
#   curl -s localhost:8778/health
#
# Weights are NOT baked in: they download on first start at the revision pinned by
# the preset, into the mounted Hugging Face cache. Needs the NVIDIA driver and
# the NVIDIA Container Toolkit on the host; CUDA runtime libraries come with the
# PyTorch wheels.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Triton (used by the flash-linear-attention kernels for Qwen3.5) compiles small
# launcher stubs at run time and needs a C compiler.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONPATH=/app/src
COPY pyproject.toml uv.lock LICENSE README.md ./
COPY vendor/jevbench ./vendor/jevbench
COPY src ./src
RUN uv sync --frozen --extra cuda --no-dev

EXPOSE 8778
# Inside the container the server binds all interfaces; publish the port to
# 127.0.0.1 on the host (as above) -- the server has no authentication.
ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "python", "-m", "ryotide.server", "--host", "0.0.0.0"]
CMD ["--preset", "ryotide-gemma"]
