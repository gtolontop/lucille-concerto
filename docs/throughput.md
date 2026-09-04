# Where the per-rack number comes from

The floor needs one number to hang everything else off: how many tokens per
second one NVL72 Rubin rack can produce for a given teacher. This is an
**estimate with stated assumptions**, not a benchmark. It lives in
`floor.js` as `rackCapacity()`.

## The chain

```
capacity = anchor
         × rubin gain
         × batch gain
         × (anchor active params / this model's active params)
         × (this model's spec gain / anchor spec gain)

capped by:  rack FP4 flops × MFU / (2 FLOP per active param per token)
```

| term | value | why |
|---|---|---|
| `anchor` | 120 k tok/s | one GB300 NVL72 at its **throughput-max** point on a 671B-A37B in FP4 with a single MTP head — not its interactive point |
| `rubin gain` | ×3.3 | Rubin NVL72 against GB300 NVL72, FP4 dense flops and HBM4 bandwidth |
| `batch gain` | ×2.0 | a dataset build has no latency SLA at all: batch is pushed until the KV cache says stop, not until the interactivity budget does |
| `active params` | ÷ ratio | decode cost per token scales with *active* parameters, not total |
| `spec gain` | E[accepted] | `(1 − a^(k+1)) / (1 − a)` for `k` MTP heads at acceptance `a` |

The anchor is the load-bearing assumption. 120 k tok/s across 72 GPUs is
~1.7 k tok/s per GPU; at 74 GFLOP per token that is well under 1 % of the
rack's dense FP4 throughput. That is not a contradiction — decode is bound by
memory traffic, not arithmetic — but it does mean this figure describes a
*serving configuration*, not a silicon limit. A different batching or
parallelism strategy moves it.

## The ceiling

The compute roofline is applied as a hard cap:

```
2 FLOP per active parameter per token, against 3.6 EF FP4 at 25 % MFU
```

For a 55B-active model that is 8.2 M tok/s per rack. The estimate lands at
708 k — 11× below — so the cap never binds here. It exists so that editing a
constant carelessly cannot produce a number the hardware could not reach.

## What it gives

| teacher | total | active | MTP | accepted/step | per rack |
|---|---|---|---|---|---|
| `concerto-6` | 1.6 T | 55 B | 4 heads @ 0.62 | 2.39 | **708 k tok/s** |
| `glm-5.3` | 744 B | 40 B | 1 head @ 0.72 | 1.72 | **700 k tok/s** |
| `glm-5.3-flash` | 320 B | 18 B | 1 head @ 0.75 | 1.75 | **1.58 M tok/s** |

`concerto-6` is fictional — it is the teacher this floor was built around.
The two GLM entries use published figures.

Flash lands well above the others because it activates 18 B parameters
instead of 55 B. It is not clamped into the same band as the large models:
that gap is the thing the model is supposed to show.

## Sources

- [Together AI — GLM-5.2: 744 B total, 40 B active, 256 experts (8 routed + 1 shared)](https://www.together.ai/models/glm-52)
- [Z.ai — GLM-5.3 shares the GLM-5.2 base model, improvements are post-training](https://docs.z.ai/guides/llm/glm-5.3)
- [Local AI Zone — GLM-5.3-Flash, 320 B-A18 B](https://local-ai-zone.github.io/blog/glm-5-3-flash-deep-dive.html)
- [Modular — GLM-5, 744 B MoE](https://www.modular.com/models/glm-5)

## Changing it

Every term is a named constant at the top of the hardware section in
`floor.js`. Change the anchor and everything downstream — sessions, samples
per second, dataset ETA — moves with it, because nothing else in the app
holds a throughput number of its own.
