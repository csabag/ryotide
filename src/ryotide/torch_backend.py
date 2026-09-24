"""PyTorch backend: the same single forward pass, on CUDA, Apple MPS or CPU.

The MLX path is the reference implementation; this one exists so RYOTIDE runs
where MLX does not (NVIDIA GPUs, Linux, WSL). Everything that decides WHAT is read
-- prompt, chat template, answer prefix, option markers, label folding -- lives in
the adapter and is shared; this class only turns token ids into the masked
next-token distribution at the final position.

No KV-cache branching here: each read is one full forward pass. With one option
order (the default configuration) that is exactly what the MLX path does too.

Memory options, for GPUs smaller than the bf16 model (Gemma 4 E4B is ~16 GB):

  quant="int8" | "nf4"   bitsandbytes weight quantization (CUDA only). It
                         quantizes Linear layers; embeddings stay in bf16.
  low_vram=True          keep Gemma 4's per-layer embedding tables (2.8 B params)
                         in system RAM and run their lookup there; audio / vision
                         towers (never used for text) stay offloaded. Measured on
                         CUDA, Gemma 4 E4B int8: running memory 12.1 -> 6.0 GB with
                         decisions identical, but the LOAD peak stays ~12 GB --
                         transformers passes offloaded weights through the GPU while
                         loading -- and system RAM peaks ~28 GB. It does not make
                         Gemma fit a 12 GB card; --quant nf4 does (10.2 GB peak).
"""
from __future__ import annotations

from typing import Sequence

# Modules worth keeping off the GPU. Names are matched as prefixes of the model's
# own module paths; anything absent from a given architecture is simply ignored.
LOW_VRAM_CPU_MODULES = (
    "model.language_model.embed_tokens_per_layer",   # Gemma 4 per-layer embeddings
    "model.audio_tower", "model.embed_audio",        # Gemma 4 audio path
    "model.vision_tower", "model.embed_vision",      # Gemma 4 / Qwen3.5 vision path
    "model.visual",                                  # Qwen-VL style vision tower
)


def _pick_device(device: str | None) -> str:
    import torch
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchClassifier:
    def __init__(self, model_id: str, revision: str | None = None,
                 device: str | None = None, dtype: str = "bfloat16",
                 quant: str | None = None, low_vram: bool = False):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = _pick_device(device)
        self.dtype = getattr(torch, dtype)
        self.quant = quant
        self.low_vram = low_vram
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        kwargs = {"revision": revision, "dtype": self.dtype}
        if quant:
            if not self.device.startswith("cuda"):
                raise ValueError(f"--quant {quant} needs CUDA (bitsandbytes); device is {self.device}")
            from transformers import BitsAndBytesConfig
            # With low_vram some modules sit on the CPU; bitsandbytes refuses that
            # unless told to keep those (unquantized) modules in system RAM.
            offload = {"llm_int8_enable_fp32_cpu_offload": True} if low_vram else {}
            kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_8bit=True, **offload) if quant == "int8" else
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=self.dtype, **offload))
        if quant or low_vram:
            kwargs["device_map"] = self._device_map(AutoConfig, AutoModelForCausalLM, model_id, revision)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(self.device).eval()
        self.offloaded = sorted({k for k, v in (getattr(self.model, "hf_device_map", None) or {}).items()
                                 if v == "cpu"})
        self.cpu_lookups = []
        if low_vram:
            self._run_embeddings_on_cpu(model_id, revision)

    # accelerate treats "cpu" in a device map as offloaded STORAGE: the weights are
    # copied to the GPU to execute. For a 5.6 GB embedding table that means a 5.6 GB
    # spike on every forward pass -- measured on CUDA: 5.3 GB resident, 11.2 GB peak.
    # An embedding is a lookup, so run it where the table lives: rebuild the module in
    # system RAM from the checkpoint, send it token ids, move only the rows back.
    CPU_LOOKUP_MODULES = ("model.language_model.embed_tokens_per_layer",)

    def _run_embeddings_on_cpu(self, model_id, revision):
        import json as _json
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        torch = self.torch
        out_dev = torch.device("cuda", 0) if self.device == "cuda" else torch.device(self.device)
        try:
            index = hf_hub_download(model_id, "model.safetensors.index.json", revision=revision)
            weight_files = _json.load(open(index))["weight_map"]
        except Exception:
            weight_files = None
        for name in self.CPU_LOOKUP_MODULES:
            if name not in self.offloaded:
                continue
            parent_name, attr = name.rsplit(".", 1)
            parent = self.model.get_submodule(parent_name)
            old = getattr(parent, attr)   # left as is: detaching its offload hook would
                                          # materialise the old table on the GPU
            key = f"{name}.weight"
            fname = weight_files[key] if weight_files else "model.safetensors"
            with safe_open(hf_hub_download(model_id, fname, revision=revision), framework="pt") as f:
                weight = f.get_tensor(key).to(self.dtype)
            new = type(old)(weight.shape[0], weight.shape[1], old.padding_idx,
                            getattr(old, "scalar_embed_scale", 1.0))
            new.weight = torch.nn.Parameter(weight, requires_grad=False)
            new.to("cpu").eval()
            inner = new.forward
            new.forward = lambda ids, _f=inner, _d=out_dev: _f(ids.to("cpu")).to(_d)
            setattr(parent, attr, new)
            del old
            self.cpu_lookups.append(name)

    def _device_map(self, AutoConfig, AutoModelForCausalLM, model_id, revision) -> dict:
        """Everything on the accelerator, except LOW_VRAM_CPU_MODULES when low_vram."""
        dev = 0 if self.device == "cuda" else self.device
        if not self.low_vram:
            return {"": dev}
        import torch
        cfg = AutoConfig.from_pretrained(model_id, revision=revision)
        with torch.device("meta"):
            skeleton = AutoModelForCausalLM.from_config(cfg)
        names = {n for n, _ in skeleton.named_modules()}
        dmap = {"": dev}
        for prefix in LOW_VRAM_CPU_MODULES:
            if prefix in names:
                dmap[prefix] = "cpu"
        del skeleton
        return dmap

    def _input_device(self):
        # With a device map the inputs go to the accelerator; accelerate's hooks move
        # them to CPU for any offloaded module and back.
        return (0 if self.device == "cuda" else self.device) if (self.quant or self.low_vram) \
            else self.device

    def read(self, tokens: Sequence[int], marker_ids: Sequence[int]) -> tuple[list[float], float]:
        """One forward pass; the next-token distribution at the LAST position.

        Returns (probabilities over the markers, renormalised) and the share of
        the full-vocabulary softmax that sat on the markers before masking.
        """
        torch = self.torch
        ids = torch.tensor([list(tokens)], device=self._input_device())
        with torch.inference_mode():
            # logits_to_keep=1: never materialise [n_tokens, vocab] logits
            out = self.model(input_ids=ids, logits_to_keep=1, use_cache=False)
        row = out.logits[0, -1].float()
        sel = torch.tensor(list(marker_ids), device=row.device)
        mass = float(torch.softmax(row, -1)[sel].sum())
        probs = torch.softmax(row[sel], -1).tolist()
        return probs, mass
