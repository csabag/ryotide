#!/bin/bash
# Reproduces bespokelabsai/nimble deploy/modal_app.py on a plain pod:
# prepare_model (CPU PEFT safe-merge of the pinned adapter) + the SGLang image's serving stack.
set -x
REV=93ec5d6ff1a9cd31d6cc0e0c58d312465d36de7c
UPSTREAM=7f84bedc169439f03379c2fa8d00ada220af2295
export HF_HOME=/models/huggingface
mkdir -p /models /opt/app/nimble/scoring
uv venv -q --python 3.12 /opt/nimble-api
uv pip install -q --python /opt/nimble-api/bin/python "openjev-sglang @ https://github.com/ekzhang/openjev-sglang/archive/$UPSTREAM.tar.gz" 'transformers==5.17.0' 'huggingface-hub==1.32.0'
uv venv -q --python 3.12 /opt/merge
uv pip install -q --python /opt/merge/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.8.0+cpu" transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 huggingface-hub==1.32.0
S=/root/src/bespokelabsai_nimble
cp $S/nimble/__init__.py $S/nimble/compat.py /opt/app/nimble/
cp $S/nimble/scoring/__init__.py $S/nimble/scoring/parallel_schema.py /opt/app/nimble/scoring/
cp -r $S/nimble/serving /opt/app/nimble/serving
/opt/merge/bin/python /root/nimble_merge.py
echo SETUP_DONE
