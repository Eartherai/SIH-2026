# Architecture decision record (Phase 2)

For every component: what we kept, what we removed, why, the evidence, and the
trade-off. Negative results are evidence and are listed as such. The governing
question throughout: *"What experiment proves this belongs here?"*

## KEPT (evidence-backed)

| Component | Evidence it belongs | Trade-off |
|---|---|---|
| **Raw input (no preprocessing)** | E06 (our chain) mAP50 0.032 and E07 (thesis 5-step) 0.043 both < raw 0.116, **matched**. Preprocessing hurts SSS. | Loses the sonar-specific conditioning that *sounds* right; QC/features still use it. |
| **Tiling** | Targets ~24 px; full-frame at network res loses them. Ablation row B (no tiling) ≈ A but tiling needed for large mosaics. | Small per-frame latency cost. |
| **Learned FP filter** | Ablation: precision 0.247→0.322, falsely-alarmed frames 37→25, keeps 19/21 TPs. Beats hand rules (which keep only 12/21). | Fitted on 30-object val — thin. |
| **Confidence calibration** | ECE improves on fit split; `calibrated:false` when unfit. | Thin fit basis. |
| **Deduplication** | Turns detections into unique hazards; √N uncertainty reduction. | Sequence mode weak without geo. |
| **Geolocation-or-refuse** | Refusal is enforced by test; error budget per fix. | Accuracy unvalidated (no nav data). |
| **Speckle-augmented training (candidate primary)** | E08 converts 0% → ~41% recall retention under speckle; better on every degradation mode. | Clean full-test mAP 0.116→0.076; undertrained. **Not yet promoted to primary** — needs a longer run to recover clean ranking. |

## REMOVED / REJECTED (with evidence)

| Component | Why rejected | Evidence |
|---|---|---|
| **Our sonar preprocessing at inference** | 12× F1 collapse mismatched; still worse matched | E06, and the mismatch table (`BENCHMARKS §3`) |
| **Thesis 5-step preprocessing on SSS** | FLS gain does not transfer to SSS | E07 matched: mAP50 0.043 < 0.116 raw |
| **SSM-DETR / structure-saliency transformer** | 276 GFLOPs — not edge-viable; and detector, not features, is our bottleneck | Thesis's own cost table; our small-object analysis shows the gap is detection recall/large-target bias, not feature saliency |
| **Autoencoder anomaly branch** | AUROC ~0.5 (chance) on small SSS targets | `experiments/anomaly_ae.json` |
| **Segmentation verifier (SEAUNet-lite)** | No mask supervision on our SSS data (MILCO/NOMBO is boxes only); AI4Shipwrecks has masks but is wrecks (large targets — exactly the class we *miss*), so it can't verify our small-target failures | dataset audit (`DATA_CARD.md`); not built rather than built on absent labels |
| **UATD in the SSS pipeline** | FLS, not SSS; Chinese lakes | `UATD_USAGE_DECISION.md` |

## DEFERRED (reasoned, not run)

| Component | Why deferred |
|---|---|
| Edge/structure features added to the FP filter | The val fit has 19 positives; a few-feature delta cannot be distinguished from noise on that basis. Needs a larger labelled val set first. |
| Feature-embedding anomaly (PaDiM/PatchCore over the backbone) | The correct replacement for the rejected AE, but a new subsystem; logged as the top anomaly path. |
| Licence-clean torchvision detector (trained) | Interface implemented + tested; training is a separate long run. Top remaining task for a government-clean path. |
| UATD auxiliary pretraining A/B | Plausible but low expected value (our bottleneck is SSS data, not init); measure before claiming. |
| Temporal tracker (DeepSORT-style) | Spatial/temporal dedup already covers the operational need; upgrade only if a continuous-survey use case demands tracks. |
| Cross-track downsampling (TR-YOLOv5s) | Genuinely SSS-aware (fixes anisotropic resolution), but geometry-changing → needs a matched retrain, and our preprocessing track shows resampling must be trained not applied at inference. Implement+evaluate as a matched experiment, not an inference knob. |
| Segmentation head (BHP-UNet/SEAUNet style) + anti-noise blending | No SSS mask labels on our data; AI4Shipwrecks masks are large-target wrecks. Revisit if a masked SSS debris set is obtained. |

## FINAL PRODUCTION ARCHITECTURE

```
RAW SSS ─▶ QC ─▶ [preprocessing OFF by default] ─▶ TILING ─▶ YOLO11n (raw-trained)
                                                                │
  REPORT ◀─ PRIORITY ◀─ GEOLOCATION(or refuse) ◀─ DEDUP ◀───────┤
                                                                │
                                CALIBRATION ◀─ LEARNED FP FILTER ┘
```

Unchanged from Phase 1 in shape — because **every Phase-2 addition either failed
to beat it (preprocessing, anomaly, segmentation) or is a training-time change to
the same detector (speckle aug)**. The Phase-2 contribution is not new boxes on
the diagram; it is *evidence* about which boxes should and should not be there.

## Trade-offs, stated

- We privilege **precision and honesty** over headline mAP: the FP filter and the
  refuse-to-geolocate behaviour cost recall/coverage but make the output trustable.
- We privilege **matched, held-out measurement** over adopting published gains:
  the thesis's FLS results, however strong, do not license an SSS claim.
- We privilege the **smallest system that survives scrutiny** over the most
  sophisticated diagram: rejected components are documented, not hidden.
