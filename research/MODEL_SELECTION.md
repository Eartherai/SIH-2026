# Model selection

## The constraints that actually decided it

1. **No pretrained sonar detector exists** for this task. Whatever we choose must
   be *trainable to convergence on a 24 GB Apple Silicon laptop* from 447 training
   objects.
2. **Targets are tiny.** Measured over all 668 annotations: mean box is
   **5.7% × 3.3%** of the frame — roughly 24×14 px on a 416 px frame. Small-object
   performance dominates everything else.
3. **Apple Silicon first.** CUDA-only or TensorRT-only paths are disqualified.
4. **Licence matters** for a government deliverable (`LEGAL_AND_LICENSES.md`).

## Candidates considered

| Candidate | Small objects | MPS | Trains on 24 GB | ONNX | Licence | Verdict |
|---|---|---|---|---|---|---|
| **YOLO11n (Ultralytics)** | Strong (P3 stride-8 head) | Yes | Yes, comfortably | Yes | **AGPL-3.0** | **Chosen** — primary + edge |
| YOLO11s/m | Stronger | Yes | Yes, slower | Yes | AGPL-3.0 | Deferred: 447 objects will not support the extra capacity |
| RT-DETR / RF-DETR | Good, but data-hungry | Partial | Marginal | Partial | Apache-2.0 / mixed | Rejected: transformer detectors need far more data than we have |
| torchvision FCOS / RetinaNet (MobileNet) | Moderate | Yes | Yes | Yes | **BSD-3** | **Implemented as an alternative backend**, not trained |
| Faster R-CNN | Moderate | Yes | Heavy | Yes | BSD-3 | Rejected: two-stage cost without an accuracy case at this data scale |
| U-Net / segmentation | Excellent localisation | Yes | Yes | Yes | Permissive | Rejected: the dataset has **boxes, not masks**. Training segmentation would require inventing masks we do not have. |

### On segmentation specifically

PS 26057 offers "bounding boxes **or** pixel-level masks". We chose boxes because
the only open, ungated, appropriately-licensed dataset we could obtain
(MILCO/NOMBO) is annotated with boxes. Deriving masks from boxes and then
reporting mask metrics would be fabricating supervision. AI4Shipwrecks provides
real masks and is the correct route to a segmentation head later — for wrecks,
which are large targets, not small debris.

## Chosen configuration

| Role | Model | Why |
|---|---|---|
| **Baseline** | YOLO11n, stock hyperparameters | Simplest thing that produces a real result — experiment `E03` |
| **Primary** | YOLO11n, sonar-domain-tuned augmentation | Best measured accuracy — experiment `E04` |
| **Edge candidate** | The same YOLO11n (2.58 M params, 6.3 GFLOPs, 16 MB) | Already small enough; ONNX export path exists |

Baseline and edge candidate are the same network. That is a real finding, not an
omission: at 2.58 M parameters YOLO11n is *already* an edge model, and shrinking
further would cost accuracy we cannot spare on 447 training objects.

## Sonar-domain augmentation policy

Chosen from the physics, not from a default config. Implemented in
`scripts/train.py` and recorded per-run in `experiments/registry.jsonl`.

| Augmentation | Setting | Justification |
|---|---|---|
| Rotation (`degrees`) | **0** | A side-scan waterfall has a *fixed* geometry: across-track is range, along-track is time. Rotating it produces an image no sonar can create, and destroys the range-dependent shadow geometry the model must learn. |
| Horizontal flip | 0.5 | Swaps port and starboard channels. Physically realisable. |
| Vertical flip | 0.5 | Reverses survey heading. Physically realisable. |
| Mosaic | 1.0 → **0.3** | Mosaic tiles four images and downscales them, which is actively harmful when targets are already ~24 px across. |
| Scale jitter | 0.5 → **0.25** | Same reasoning: aggressive downscaling erases small targets. |
| Hue / saturation | **0** | Meaningless on single-channel acoustic data. |
| Value (brightness) | 0.4 | Physically meaningful — stands in for sonar gain variation between surveys. |

## Two training-stability findings worth recording

**1. Ultralytics 8.4.130 diverged; 8.3.253 did not.** On the initial run,
`val/cls_loss` climbed to ~1.1 × 10⁶ while `train/cls_loss` stayed flat. Disabling
AMP did not fix it. Pinning to the stable 8.3 line did. `requirements.txt`
therefore pins `ultralytics>=8.3,<8.4`.

**2. AMP is disabled by default on MPS.** Mixed precision is off unless `--amp` is
passed. Given finding 1 we cannot claim AMP was the *cause* of that particular
divergence, only that we do not enable it on this backend.

## What we did not do

- No comparison against a *trained* torchvision baseline. The backend is
  implemented and interface-tested; it has not been trained, so no accuracy claim
  is made for it.
- No YOLO11s/m sweep. With 447 training objects the larger variants were judged
  unlikely to help, but this was **not measured**.
- No hyperparameter search. E03 → E04 is a single reasoned step, not a sweep.
