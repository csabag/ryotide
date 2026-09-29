# Decision Index runs (apolinario/decision-index)

Helpers used for the self-scored Decision Index 0.2.1 run of RYOTIDE-Qwen v0.3.1 (36.63;
apolinario/decision-index#28). Run them inside a decision-index checkout (kit commit `87d4650`),
next to one RYOTIDE server per shard.

- `di_shards.py` -- `run` shard K of N (a request belongs to shard `crc32(run_id) % N`;
  `--skip-done` re-splits a model that is already running), then `merge` the shards and score once
  with the kit's own `score_run`
- `kernel_ab.py` -- send the same suite rows to two servers (alternating order) and compare speed
  and answers

```bash
python -m ryotide.server --preset ryotide-qwen --port 8002 &     # one server per shard
python di_shards.py run --shard 0 --of 3 --port 8002 --model ryotide-qwen --out runs/rq/shard0
python di_shards.py merge --out runs/RYOTIDE-Qwen runs/rq/shard0 runs/rq/shard1 runs/rq/shard2
```

Lessons: do not start extra servers on the GPU during a run (long ToolRet requests, up to 200
questions, need the headroom); the prebuilt `kernels-community/causal-conv1d` Hub kernel gave only
about 10% on this workload.
