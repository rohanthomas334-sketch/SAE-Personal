# SAE Feature-Transfer Experiment — Artifact Manifest

This folder contains the outputs of an experiment testing whether mathematical
reasoning can be transferred from a larger model (Gemma‑2‑9B) into a smaller one
(Gemma‑2‑2B) by "stitching" between their sparse‑autoencoder (SAE) feature spaces,
optionally constrained to an "oracle" direction derived from instruction tuning.

**Headline finding: it does not transfer.** Across every training objective and
ablation, grafting the 9B's (translated) representation into the 2B never beat the
2B's own baseline, and at high strength it actively degraded it. Two stacked causes
were identified: (1) the SAE reconstruction itself loses reasoning‑relevant
information, and (2) the cross‑model translation carries no usable reasoning either.

---

## Quick results summary

| Condition | GSM8K accuracy | Meaning |
|---|---|---|
| Base Gemma‑2‑2B (no graft) | **0.260** | the student's own baseline |
| Gemma‑2‑2B‑**it** (north star) | **0.535** | the gap that "should" be recoverable |
| SAE stitch (9B→2B via SAEs) | ≤ baseline; **0.085** at full strength | never transfers; collapses when leaned on |
| Oracle‑projected stitch | ≈ baseline (~0.26–0.29) | damage control only |
| **Random**‑basis projection | ≈ same as oracle | projection = rank‑regularization, *not* the IT direction |
| Oracle‑alone (no 9B) | ≈ baseline | the IT direction has no standalone lift |
| SAE round‑trip (2B's own act through its own SAE, no 9B) | 0.27 → **0.156** | SAE reconstruction alone destroys reasoning |
| Raw stitch (9B→2B, **no SAE**) | 0.26 → **0.150** | ≈ round‑trip → translation carries no reasoning either |

Interpretability: the top oracle direction (PC0) aligns ~0.84–0.89 with a single
SAE feature — far above random (~0.11) — so the instruction‑tuning direction is
*feature‑localized* (though distributed in variance: PC0 ≈ 9% after removing the
BOS/attention‑sink token). All accuracy numbers are on the first 200 GSM8K‑test
problems unless noted (the round‑trip curve in `results.json` is n=500).

---

## File-by-file

### `SAE_Negative_Case_Result.ipynb`  (~0.5 MB)
The complete notebook: all code, all cell outputs (training logs, eval numbers),
and the plots. **This is the master record and the source of truth.** The two
stitch classes (`BottleneckStitch`, `RawResidualStitch`) and the eval/oracle
functions live here — you need their class definitions to load the `.pth` files.

### `bottleneck_stitch.pth`  (~17 MB)
Trained weights (`state_dict`) of the **SAE‑feature stitch** — the main method.
It maps 9B SAE features → 2B SAE features through a 64‑dim bottleneck
(`dim_a = dim_b = 16384`, `bottleneck_dim = 64`). Load with the `BottleneckStitch`
class from the notebook:
```python
import torch
stitch = BottleneckStitch(16384, 16384, bottleneck_dim=64)   # class is defined in the notebook
stitch.load_state_dict(torch.load('bottleneck_stitch.pth', map_location='cpu'))
```

### `raw_residual_stitch.pth`  (~48 MB)
Trained weights (`state_dict`) of the **no‑SAE control stitch**. A plain MLP that
maps the 9B residual stream directly to the 2B residual stream
(`Linear(3584, 2048) → GELU → Linear(2048, 2304)`), bypassing both SAEs. Class:
```python
import torch, torch.nn as nn
class RawResidualStitch(nn.Module):
    def __init__(self, d_b=3584, d_a=2304, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_b, hidden), nn.GELU(), nn.Linear(hidden, d_a))
    def forward(self, x):
        return self.net(x)

raw = RawResidualStitch()
raw.load_state_dict(torch.load('raw_residual_stitch.pth', map_location='cpu'))
```

### `oracle_pca.pt`  (~21 MB)
A dict with the **oracle basis**: PCA of the (Gemma‑2‑2b‑**it** minus Gemma‑2‑2b base)
residual‑stream deltas at layer 20, collected on GSM8K‑train prompts.
```python
import torch
d = torch.load('oracle_pca.pt', map_location='cpu')
pcs  = d['principal_components']   # [2304, 2304], columns are PC directions
ev   = d['explained_var']          # variance explained per component
```
Check which version this is: `ev[0] ≈ 0.086` → the **cleaned** (BOS/sink token
excluded) collection; `ev[0] ≈ 0.18` → the original (BOS included). The cleaned
version is the one to report.

### `oracle_projected_comparison.png`  (~145 KB)
Plot of the eval sweep: GSM8K accuracy vs graft strength for the baseline and the
oracle‑projected stitch at several ranks. Shows projection ≈ baseline (no transfer).

### `oracle_vs_random_cosine.png`  (~47 KB)
Histogram of how strongly random residual directions align (max cosine) with any
SAE feature, with the oracle PCs marked — the oracle PCs sit far to the right of
the random distribution (the feature‑localization result).

### `results.json`  (~3 KB)
All eval accuracy numbers as nested dicts:
```python
import json
r = json.load(open('results.json'))
# r['all_results'] : main sweep + random-basis ablation + oracle-alone,
#                    keyed like 'baseline_s0.5', 'proj_r16_s0.3', 'random_r8_s1.0', ...
# r['raw_results'] : {strength: accuracy} for the raw (no-SAE) stitch
# r['rt_curve']    : {strength: accuracy} for the SAE round-trip control (n=500)
```

---

## To reproduce / reload everything

1. Re‑download the base models and SAEs from HuggingFace (they were **not** saved
   here — they're large and re‑downloadable): `google/gemma-2-2b`,
   `google/gemma-2-9b`, and the Gemma‑Scope canonical residual SAEs
   `gemma-scope-2b-pt-res-canonical` (layer 20, width 16k) and
   `gemma-scope-9b-pt-res-canonical` (layer 30, width 16k).
2. Open the notebook to get the class/function definitions, then load the `.pth`/
   `.pt` files as shown above.

## Caveats for write-up
- Sweep/ablation accuracies are n=200; the round‑trip curve is n=500. For final
  figures, rerun the headline comparisons (baseline / best stitch / random /
  round‑trip) on the full 1,319‑problem test set with a couple of seeds.
- The oracle PCs are directions of *variance* in the IT effect; PC0's variance
  share dropped (~0.18 → ~0.086) once the BOS/sink token was excluded, but its
  feature alignment survived — so the localization is genuine, not a sink artifact.
