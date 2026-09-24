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
  low_vram=True          keep modules that are cheap to run from system RAM on the
                         CPU: Gemma 4's per-layer embedding tables (2.8 B params,
                         a lookup of a few rows per token) and its audio / vision
                         towers (never used for text). Measured on Gemma 4 E4B:
                         int8 alone ~10.8 GB of weights, int8 + low_vram ~5-6 GB.
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
            kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_8bit=True) if quant == "int8" else
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=self.dtype))
        if quant or low_vram:
            kwargs["device_map"] = self._device_map(AutoConfig, AutoModelForCausalLM, model_id, revision)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(self.device).eval()
        self.offloaded = sorted({k for k, v in (getattr(self.model, "hf_device_map", None) or {}).items()
                                 if v == "cpu"})

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
