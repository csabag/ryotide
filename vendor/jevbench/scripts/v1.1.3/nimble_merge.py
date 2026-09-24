from pathlib import Path
MODEL = "bespokelabs/Bespoke-Nimble-9B"
REVISION = "93ec5d6ff1a9cd31d6cc0e0c58d312465d36de7c"
MERGED = f"/models/openjeff-{REVISION}"
def prepare_model():
    import hashlib
    import json
    import os
    import shutil
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    output = Path(MERGED)
    if (output / "READY.json").exists():
        print(str(output))
    adapter = Path(snapshot_download(MODEL, revision=REVISION))
    contract = json.loads((adapter / "schema_config.json").read_text())
    prompt_hash = hashlib.sha256((adapter / "parallel_schema.py").read_bytes()).hexdigest()
    if prompt_hash != contract["prompt_code_sha256"]:
        raise ValueError("Published prompt does not match the training contract")
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        contract["model"], revision=contract["revision"], dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="sdpa", trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload(safe_merge=True)
    temporary = Path(str(output) + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(adapter).save_pretrained(temporary)
    shutil.copy(adapter / "schema_config.json", temporary)
    (temporary / "READY.json").write_text(json.dumps({
        "model": MODEL, "revision": REVISION, "base": contract["model"],
        "base_revision": contract["revision"], "prompt_sha256": prompt_hash,
        "precision": "bfloat16", "merge": "PEFT safe_merge",
    }, indent=2))
    os.rename(temporary, output)
    print(str(output))

prepare_model()
