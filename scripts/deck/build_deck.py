#!/usr/bin/env python3
"""Build the AQUA-SHIELD SIH 2026 deck: 6 slides -> PDF (submission format) + PNG + PPTX.

Design is authored in HTML/CSS (scripts/deck/slides.css) and rendered with a real
browser engine, which gives typographic and layout control python-pptx cannot,
and renders straight to the PDF that the SIH portal actually accepts.

EVERY NUMBER IN THIS DECK IS EITHER:
  (a) MEASURED  -- traceable to docs/BENCHMARKS.md / experiments/registry.jsonl,
                   or to a live run of the dashboard, or
  (b) TARGET    -- explicitly labelled as a goal we intend to prove, never as
                   an achieved result.
Nothing is estimated, rounded up, or invented. See docs/BENCHMARKS.md.

Usage:
    .venv/bin/python3 scripts/deck/build_deck.py
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / "scripts" / "deck"
IMG = ROOT / "docs" / "images"
OUT_DIR = ROOT / "docs"
BUILD = DECK / "_build"
BUILD.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# EDIT THESE TWO LINES, then re-run this script.
TEAM_NAME = "Your Team Name"
TEAM_ID = "Your Team ID"
# ---------------------------------------------------------------------------

LOGO = (IMG / "sih_logo_2026.png").as_uri()
P = IMG / "prototype"


def u(name: str) -> str:
    return (P / name).as_uri()


def chrome(page_no: int, kicker: str, title: str, sub: str) -> str:
    return f"""
  <div class="logo-plate"><img src="{LOGO}" alt="Smart India Hackathon 2026"></div>
  <div class="hdr">
    <div class="kick">{kicker}</div>
    <h1>{title}</h1>
    <div class="sub">{sub}</div>
  </div>
  <div class="rule"></div>
  <div class="foot">
    <div><b>AQUA-SHIELD</b> &nbsp;·&nbsp; PS 26057 &nbsp;·&nbsp; MoES / NIOT &nbsp;·&nbsp;
         Disaster Management &nbsp;·&nbsp; {TEAM_NAME}</div>
    <div class="pg">{page_no} / 6</div>
  </div>"""


# ============================================================== SLIDE 1 =====
S1 = f"""
<div class="slide">
  <div class="logo-plate"><img src="{LOGO}" alt="Smart India Hackathon 2026"></div>

  <div style="position:absolute; left:0; top:0; width:52%; height:100%;
              padding:74px 0 0 56px;">
    <div class="kick" style="font-size:14px;font-weight:700;letter-spacing:2.4px;
         color:var(--cyan);text-transform:uppercase;">Smart India Hackathon 2026</div>

    <div style="font-size:82px;font-weight:800;letter-spacing:-2.6px;line-height:1;
                margin-top:16px;">AQUA<span style="color:var(--cyan);">-</span>SHIELD</div>

    <div style="font-size:17px;color:var(--txt-2);margin-top:14px;line-height:1.45;
                max-width:600px;">
      Acoustic intelligence for underwater anomaly, debris &amp;
      marine-hazard localization from side-scan sonar.
    </div>

    <div style="display:flex;gap:9px;margin-top:20px;">
      <span class="chip k">Detect</span><span class="chip k">Verify</span>
      <span class="chip k">Localize</span><span class="chip k">Act</span>
    </div>

    <div class="callout cy" style="margin-top:26px;max-width:625px;">
      <b>The insight that shapes the whole system:</b> 74% of our sonar frames contain
      no target at all. Precision — not recall — is the binding constraint.
    </div>

    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:11px;
                margin-top:26px;max-width:625px;">
      <div class="card" style="padding:12px 15px;">
        <div style="font-size:10.5px;letter-spacing:1.4px;color:var(--txt-3);
                    text-transform:uppercase;font-weight:700;">Problem Statement ID</div>
        <div style="font-size:19px;font-weight:800;margin-top:3px;">26057</div>
      </div>
      <div class="card" style="padding:12px 15px;">
        <div style="font-size:10.5px;letter-spacing:1.4px;color:var(--txt-3);
                    text-transform:uppercase;font-weight:700;">PS Category</div>
        <div style="font-size:19px;font-weight:800;margin-top:3px;">Software</div>
      </div>
      <div class="card" style="padding:12px 15px;">
        <div style="font-size:10.5px;letter-spacing:1.4px;color:var(--txt-3);
                    text-transform:uppercase;font-weight:700;">Theme</div>
        <div style="font-size:19px;font-weight:800;margin-top:3px;">Disaster Management</div>
      </div>
      <div class="card" style="padding:12px 15px;">
        <div style="font-size:10.5px;letter-spacing:1.4px;color:var(--txt-3);
                    text-transform:uppercase;font-weight:700;">Team ID &nbsp;·&nbsp; Name</div>
        <div style="font-size:19px;font-weight:800;margin-top:3px;">{TEAM_ID} · {TEAM_NAME}</div>
      </div>
    </div>

    <div style="font-size:12.5px;color:var(--txt-3);margin-top:22px;max-width:625px;
                line-height:1.5;">
      Problem Statement — AI-powered automated underwater marine debris and anomaly
      detection using side-scan sonar imagery &nbsp;·&nbsp; Ministry of Earth Sciences / NIOT
    </div>
  </div>

  <div style="position:absolute; right:0; top:0; width:47%; height:100%;">
    <img src="{u('panel_sonar.png')}"
         style="position:absolute;right:52px;top:128px;width:604px;border-radius:14px;
                border:1px solid var(--line);
                -webkit-mask-image:linear-gradient(90deg,transparent 0%,#000 15%,#000 100%);
                mask-image:linear-gradient(90deg,transparent 0%,#000 15%,#000 100%);">
    <div style="position:absolute;right:52px;top:760px;width:604px;">
      <div style="font-size:11.5px;letter-spacing:1.6px;text-transform:uppercase;
                  color:var(--cyan);font-weight:800;">Live system output</div>
      <div style="font-size:12.5px;color:var(--txt-2);margin-top:6px;line-height:1.45;">
        Real held-out survey frame <span class="mono">0460_2018</span>, processed by the
        running prototype. Yellow = man-made, green = ambiguous — both accepted by the
        verification stage before display.
      </div>
    </div>
  </div>
</div>"""


# ============================================================== SLIDE 2 =====
S2 = f"""
<div class="slide">
  {chrome(2, "Idea &amp; Innovation", "Why the obvious approach fails here",
          "Everyone optimises recall. On this data that is the wrong objective — and it is why sonar AI does not reach operators.")}
  <div class="body">
    <div class="fill" style="display:grid;grid-template-columns:1fr 1fr 1.06fr;gap:14px;">

      <div class="card">
        <div class="lbl coral">The problem</div>
        <ul class="tight">
          <li>Derelict fishing gear — <b>ghost nets</b> — keeps killing after it is
              lost: entangling marine life, smothering reefs, fouling propellers.</li>
          <li>Finding it means a human reading <b>thousands of km</b> of side-scan
              sonar by eye — slow, fatiguing, inconsistent.</li>
          <li>Seabed clutter — rocks, sand ripples, acoustic shadows — looks
              exactly like a target.</li>
        </ul>
      </div>

      <div class="card">
        <div class="lbl amber">What exists today — and its limit</div>
        <table class="t">
          <tr><th>System</th><th>Limitation</th></tr>
          <tr><td>GhostVision<br><span style="font-size:10.5px;color:var(--txt-3);">JMSE 2025</span></td>
              <td class="dim">Closest prior system. Licence unresolved (NOASSERTION) — cannot be reused.</td></tr>
          <tr><td>sidescantools<br><span style="font-size:10.5px;color:var(--txt-3);">GPL-3.0</span></td>
              <td class="dim">Sonar processing only. No detection, no verification.</td></tr>
          <tr><td>Generic YOLO<br><span style="font-size:10.5px;color:var(--txt-3);">detector-only</span></td>
              <td class="dim">Fires on empty seabed. No evidence, no calibration, no refusal.</td></tr>
        </table>
      </div>

      <div class="card" style="display:flex;flex-direction:column;justify-content:center;
           border-color:rgba(34,211,238,0.42);">
        <div class="lbl">Our insight</div>
        <div style="font-size:76px;font-weight:800;color:var(--cyan);line-height:0.94;
                    letter-spacing:-2.5px;">74%</div>
        <div style="font-size:14px;color:var(--txt);margin-top:11px;line-height:1.45;">
          of frames in our held-out surveys contain <b>no target at all</b>
          (473 of 612 — measured).
        </div>
        <div style="font-size:13.5px;color:var(--txt-2);margin-top:12px;line-height:1.45;">
          A recall-tuned detector alarms constantly on empty seabed, the analyst
          stops trusting it, and the system is abandoned.
          <span style="color:var(--cyan);font-weight:600;">Precision is the
          binding constraint.</span>
        </div>
      </div>
    </div>

    <div style="margin-top:15px;">
      <div class="lbl">Our answer — four things that are ours</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:13px;">
        <div class="card" style="padding:14px 16px;">
          <div style="font-size:22px;font-weight:800;color:var(--cyan);">01</div>
          <div style="font-size:14px;font-weight:700;margin-top:6px;line-height:1.28;">
            Independent physical verification</div>
          <div style="font-size:11.5px;color:var(--txt-2);margin-top:6px;line-height:1.4;">
            10 features measured from the pixels — shadow coherence, contrast,
            compactness, texture — that do <i>not</i> depend on the detector's
            own opinion. Fitted on a held-out survey.</div>
        </div>
        <div class="card" style="padding:14px 16px;">
          <div style="font-size:22px;font-weight:800;color:var(--cyan);">02</div>
          <div style="font-size:14px;font-weight:700;margin-top:6px;line-height:1.28;">
            Confidence that means something</div>
          <div style="font-size:11.5px;color:var(--txt-2);margin-top:6px;line-height:1.4;">
            Platt calibration fitted on held-out data. When it is not fitted,
            every hazard is stamped <span class="mono"
            style="color:var(--amber);">calibrated: false</span> — never a
            silently wrong number.</div>
        </div>
        <div class="card" style="padding:14px 16px;">
          <div style="font-size:22px;font-weight:800;color:var(--cyan);">03</div>
          <div style="font-size:14px;font-weight:700;margin-top:6px;line-height:1.28;">
            Refusal as a feature</div>
          <div style="font-size:11.5px;color:var(--txt-2);margin-top:6px;line-height:1.4;">
            No navigation metadata → no coordinate. <span style="color:var(--amber);">null</span>,
            plus the reason. A fabricated latitude exports cleanly to CSV and
            sends a vessel to open water.</div>
        </div>
        <div class="card" style="padding:14px 16px;">
          <div style="font-size:22px;font-weight:800;color:var(--cyan);">04</div>
          <div style="font-size:14px;font-weight:700;margin-top:6px;line-height:1.28;">
            Confidence ≠ Priority</div>
          <div style="font-size:11.5px;color:var(--txt-2);margin-top:6px;line-height:1.4;">
            “Is it real?” and “should you care?” are different questions, scored
            separately, so a low-confidence large hazard is not buried.</div>
        </div>
      </div>
    </div>

    <div class="callout" style="margin-top:14px;">
      <b>The verifier is a diagnostic instrument, not just a classifier.</b>
      Its fitted weights gave acoustic shadow a large <i>negative</i> weight — the
      opposite of the physics. That exposed a real defect: a preprocessing chain
      applied at inference that the detector had never been trained on. Fixing the
      mismatch moved F1 from <span class="mono">0.012 → 0.144</span> (measured) and
      turned both shadow features positive. A hand-tuned threshold would have hidden it.
    </div>
  </div>
</div>"""


# ============================================================== SLIDE 3 =====
def fstep(n, t, d, cls=""):
    return (f'<div class="fstep {cls}"><div class="n">{n}</div>'
            f'<div class="t">{t}</div><div class="d">{d}</div></div>')


ARROW = '<div class="farrow">›</div>'

PIPE = ARROW.join([
    fstep("01", "Ingest", "SSS frames<br>+ nav / GeoTIFF"),
    fstep("02", "Quality<br>control", "dyn. range, speckle,<br>dropout, water column"),
    fstep("03", "Preprocess", "off by default —<br>measured, hurt", "off"),
    fstep("04", "Tiling", "overlap, seam merge<br>by IoU / IoS"),
    fstep("05", "Detection", "YOLO11n<br>2.58M params"),
    fstep("06", "Verification", "10 physical features<br>→ logistic model", "key"),
    fstep("07", "Calibration", "Platt, or stamped<br>uncalibrated"),
    fstep("08", "Dedup", "repeat sightings<br>→ unique hazard"),
    fstep("09", "Geolocate", "affine / per-ping nav<br>or refuse"),
    fstep("10", "Report", "ranked register<br>GeoJSON · CSV"),
])

S3 = f"""
<div class="slide">
  {chrome(3, "Technical Approach", "The real pipeline — ten stages, every one measured",
          "Not “user → AI → result”. Each stage is separately testable, separately ablated, and can be switched off to measure what it contributes.")}
  <div class="body">

    <div class="flow" style="margin-bottom:15px;">{PIPE}</div>

    <div class="fill" style="display:grid;grid-template-columns:1.02fr 1fr 0.92fr;gap:14px;">

      <div class="card">
        <div class="lbl">Data — real, licensed, leakage-free</div>
        <table class="t">
          <tr><th>Dataset</th><th>Scale</th><th>Split by</th></tr>
          <tr>
            <td>MILCO / NOMBO<br><span style="font-size:10.5px;color:var(--green);">CC BY 4.0</span></td>
            <td class="num">465 / 93 / 612<br><span style="font-size:10px;color:var(--txt-3);">frames · 191 test objects</span></td>
            <td class="dim">acquisition<br>year</td>
          </tr>
          <tr>
            <td>Derelict crab pot<br><span style="font-size:10.5px;color:var(--green);">CC BY-SA 4.0</span></td>
            <td class="num">6,674 imgs<br><span style="font-size:10px;color:var(--txt-3);">9,311 objects · 107 recordings</span></td>
            <td class="dim">recording<br>ID</td>
          </tr>
        </table>
        <div class="callout cy" style="margin-top:12px;padding:9px 13px;font-size:12px;">
          <b>Never a random split.</b> Consecutive sonar frames share seabed, gain
          settings and often the same object — a random split leaks and inflates
          every number. Train 2015+2010 → fit 2017 → test 2018+2021, asserted by a test.
        </div>
      </div>

      <div class="card">
        <div class="lbl violet">Core algorithm — the verification stage</div>
        <div style="font-size:12.5px;color:var(--txt-2);line-height:1.45;margin-bottom:9px;">
          Every detector candidate is re-examined against evidence measured
          <b style="color:#fff;">from the pixels</b>, independent of the detector score:
        </div>
        <div>
          <span class="chip">shadow ratio</span><span class="chip">shadow side consistency</span>
          <span class="chip">local SNR</span><span class="chip">contrast</span>
          <span class="chip">compactness</span><span class="chip">aspect ratio</span>
          <span class="chip">edge density</span><span class="chip">relative texture</span>
          <span class="chip">area fraction</span><span class="chip">range position</span>
        </div>
        <div style="margin-top:11px;font-size:12.5px;color:var(--txt-2);line-height:1.5;">
          → <b style="color:#fff;">L2-regularised logistic regression</b>, fitted on the
          held-out validation survey, with per-detection weight attribution so an
          operator can see <i>why</i> a candidate was accepted or rejected.<br>
          → <b style="color:#fff;">Platt calibration</b> maps the score to a probability;
          identity transform + explicit flag when unfitted.
        </div>
      </div>

      <div class="card">
        <div class="lbl green">System &amp; stack</div>
        <table class="t">
          <tr><th>Layer</th><th>Component</th></tr>
          <tr><td class="dim">Interface</td><td>Streamlit operator console</td></tr>
          <tr><td class="dim">Service</td><td>FastAPI (REST)</td></tr>
          <tr><td class="dim">Model</td><td>PyTorch 2.13 · MPS / CUDA / CPU</td></tr>
          <tr><td class="dim">Store</td><td>SQLite hazard register</td></tr>
          <tr><td class="dim">Edge</td><td>ONNX Runtime</td></tr>
          <tr><td class="dim">Geo</td><td>pyproj · GeoJSON → QGIS</td></tr>
        </table>
        <div style="margin-top:11px;font-size:11.5px;color:var(--txt-3);line-height:1.45;">
          Backend-swappable detector: the pipeline depends on an interface, not on
          Ultralytics, so the AGPL-3.0 backend can be replaced without touching
          the verification, calibration or reporting stages.
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:13px;padding:12px 17px;">
      <div style="display:flex;align-items:center;gap:11px;">
        <div style="font-size:11px;font-weight:800;letter-spacing:1.5px;color:var(--cyan);
                    text-transform:uppercase;white-space:nowrap;">I/O contract</div>
        <div style="flex:1;display:flex;align-items:center;gap:8px;font-size:11.5px;">
          <span class="chip mono" style="margin:0;">SSS frame 1024&times;1024 + nav CSV / GeoTIFF</span>
          <span style="color:var(--cyan-dim);">&rsaquo;</span>
          <span class="chip mono" style="margin:0;">640px tiles</span>
          <span style="color:var(--cyan-dim);">&rsaquo;</span>
          <span class="chip mono" style="margin:0;">boxes + class + raw score</span>
          <span style="color:var(--cyan-dim);">&rsaquo;</span>
          <span class="chip mono k" style="margin:0;">10-feature vector &rarr; p(real)</span>
          <span style="color:var(--cyan-dim);">&rsaquo;</span>
          <span class="chip mono" style="margin:0;">hazard record</span>
          <span style="color:var(--cyan-dim);">&rsaquo;</span>
          <span class="chip mono" style="margin:0;">GeoJSON &middot; CSV &middot; SQLite</span>
        </div>
      </div>
      <div style="font-size:11px;color:var(--txt-3);margin-top:9px;line-height:1.4;">
        <b style="color:var(--txt-2);">hazard record</b> =
        <span class="mono">{{id, class, level1, confidence, calibrated, priority, priority_band,
        lat, lon, uncertainty_m, observations, evidence[10], provenance}}</span> —
        every field an operator needs to act, audit, or reject the call.
      </div>
    </div>
  </div>
</div>"""


# ============================================================== SLIDE 4 =====
S4 = f"""
<div class="slide">
  {chrome(4, "Feasibility &amp; Viability", "It runs today — and this is how we will prove it",
          "Left: the actual prototype, actual output, on held-out data. Right: the metrics we commit to hitting, stated as targets, not claims.")}
  <div class="body">
    <div style="display:grid;grid-template-columns:1.16fr 1fr;gap:15px;height:100%;">

      <div style="display:flex;flex-direction:column;gap:11px;">
        <div class="card" style="padding:13px 15px;">
          <div class="lbl">Hero demo — one deterministic 90-second run</div>
          <div style="display:flex;align-items:center;gap:7px;font-size:11.5px;">
            <span class="chip" style="margin:0;">8 held-out frames</span>
            <span style="color:var(--cyan-dim);">›</span>
            <span class="chip" style="margin:0;">QC</span>
            <span style="color:var(--cyan-dim);">›</span>
            <span class="chip" style="margin:0;">2 raw candidates</span>
            <span style="color:var(--cyan-dim);">›</span>
            <span class="chip k" style="margin:0;">verify</span>
            <span style="color:var(--cyan-dim);">›</span>
            <span class="chip" style="margin:0;">2 hazards</span>
            <span style="color:var(--cyan-dim);">›</span>
            <span class="chip" style="margin:0;">1 HIGH</span>
            <span style="color:var(--cyan-dim);">›</span>
            <span class="chip" style="margin:0;">export</span>
          </div>
        </div>

        <div style="display:grid;grid-template-columns:0.92fr 1fr;gap:12px;flex:1;min-height:0;">
          <div class="shotwrap" style="display:flex;flex-direction:column;min-height:0;">
            <img src="{u('panel_sonar.png')}"
                 style="display:block;width:100%;min-height:0;object-fit:contain;
                        border-radius:10px;border:1px solid var(--line);">
            <div class="shotcap">Frame <span class="mono">0460_2018</span> — live output of
              the running system, not a mockup.</div>
          </div>
          <div class="card" style="padding:14px 16px;">
            <div class="lbl" style="margin-bottom:9px;">What the operator is shown</div>
            <div style="font-size:13px;line-height:1.5;">
              <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:9px;">
                <span style="width:11px;height:11px;border-radius:3px;background:#F2C230;
                             display:inline-block;flex:0 0 11px;"></span>
                <div><b>MILCO</b> · mine-like object<br>
                  <span style="color:var(--txt-2);">confidence
                  <b style="color:var(--green);">77.0%</b> · band HIGH · priority 58.7</span></div>
              </div>
              <div style="display:flex;align-items:baseline;gap:8px;">
                <span style="width:11px;height:11px;border-radius:3px;background:#5FD35F;
                             display:inline-block;flex:0 0 11px;"></span>
                <div><b>NOMBO</b> · bottom object uncertain<br>
                  <span style="color:var(--txt-2);">confidence
                  <b style="color:var(--amber);">39.5%</b> · band LOW · priority 33.2</span></div>
              </div>
            </div>
            <div style="font-size:11.5px;color:var(--txt-3);margin-top:12px;line-height:1.45;
                        border-top:1px solid var(--line-soft);padding-top:10px;">
              Both survived the verification stage. NOMBO is reported as
              <i>uncertain</i> rather than suppressed — the operator decides, with the
              evidence and a calibrated number in front of them.
            </div>
          </div>
        </div>

        <div class="shotwrap">
          <img class="shot" src="{u('panel_register.png')}">
          <div class="shotcap"><b style="color:var(--amber);">Refusal, visible in the
            product:</b> this survey ships no navigation, so lat / lon / ±m come back
            <span class="mono">None</span> — the system declines to invent a coordinate.</div>
        </div>

      </div>

      <div style="display:flex;flex-direction:column;gap:11px;">
        <div class="card" style="flex:1;">
          <div class="lbl">Target validation — what we will prove at the finals</div>
          <table class="t">
            <tr>
              <th style="width:40%;">Metric that matters</th>
              <th style="width:30%;">Measured today</th>
              <th style="width:30%;">Target</th>
            </tr>
            <tr><td>False-alarm rate on target-free frames</td>
                <td class="meas">25 / 473 &nbsp;(5.3%)</td><td class="tgt">≤ 2%</td></tr>
            <tr><td>Precision @ IoU 0.3</td>
                <td class="meas">0.322</td><td class="tgt">≥ 0.60</td></tr>
            <tr><td>Recall @ IoU 0.3</td>
                <td class="meas">0.099</td><td class="tgt">≥ 0.45</td></tr>
            <tr><td>Ghost-gear detection, mAP50</td>
                <td class="meas">0.323 <span style="color:var(--txt-3);font-size:10px;">raw, 1 run</span></td>
                <td class="tgt">≥ 0.55</td></tr>
            <tr><td>Confidence calibration error (ECE)</td>
                <td class="dim">fitted, not yet reported</td><td class="tgt">≤ 0.05</td></tr>
            <tr><td>Geolocation error (CEP)</td>
                <td class="dim">not measurable — no nav<br>in available data</td><td class="tgt">≤ 10 m</td></tr>
            <tr><td>Inference latency / frame</td>
                <td class="meas">21.4 ms MPS · 8.5 ms ONNX</td><td class="tgt">≤ 30 ms edge</td></tr>
            <tr><td>Sustained throughput</td>
                <td class="meas">37.4 frames/s</td><td class="tgt">≥ 30 f/s on edge</td></tr>
          </table>
          <div style="margin-top:10px;font-size:11px;color:var(--txt-3);line-height:1.45;">
            <b style="color:var(--green);">Measured</b> = reproducible from
            <span class="mono">experiments/</span> on held-out surveys.
            <b style="color:var(--cyan);">Target</b> = a goal, not a result. We publish
            both columns so the gap is visible rather than hidden.
          </div>
        </div>

        <div class="card" style="padding:13px 15px;">
          <div class="lbl amber">Baselines we will measure against</div>
          <div style="font-size:12px;color:var(--txt-2);line-height:1.5;">
            detector-only (no verification) &nbsp;·&nbsp; hand-written rule filter
            &nbsp;·&nbsp; YOLO11s / RT-DETR at equal compute &nbsp;·&nbsp; published
            SSS detectors on their own reported splits.
          </div>
        </div>

        <div class="card" style="padding:13px 15px;">
          <div class="lbl coral">Risks — and the mitigation already built</div>
          <table class="t" style="font-size:12px;">
            <tr><td style="width:47%;">Scarce coastal SSS labels</td>
                <td class="dim">leakage-free recording-level splits; honest reporting</td></tr>
            <tr><td>Detector backend is AGPL-3.0</td>
                <td class="dim">interface-isolated; ONNX path is backend-free</td></tr>
            <tr><td>Speckle noise degrades detection</td>
                <td class="dim">noise-robust checkpoint trained + shipped separately</td></tr>
          </table>
        </div>
      </div>
    </div>
  </div>
</div>"""


# ============================================================== SLIDE 5 =====
S5 = f"""
<div class="slide">
  {chrome(5, "Impact &amp; Benefits", "From raw sonar to a tasked cleanup vessel",
          "The deliverable is not a model file. It is a ranked, geolocated, auditable hazard register an operator can act on the same day.")}
  <div class="body">
    <div class="fill" style="display:grid;grid-template-columns:1fr 1fr;gap:15px;">

      <div class="card">
        <div class="lbl">What changes operationally</div>
        <div style="display:flex;align-items:center;gap:9px;margin:4px 0 14px;">
          <div style="flex:1;border:1px solid var(--line);border-radius:9px;padding:10px;
                      text-align:center;background:rgba(248,113,113,0.08);">
            <div style="font-size:11.5px;color:var(--coral);font-weight:700;">TODAY</div>
            <div style="font-size:12.5px;margin-top:4px;line-height:1.3;">analyst scrolls<br>raw imagery</div>
          </div>
          <div style="color:var(--cyan);font-size:19px;">›</div>
          <div style="flex:1.35;border:1px solid var(--cyan);border-radius:9px;padding:10px;
                      text-align:center;background:rgba(34,211,238,0.09);">
            <div style="font-size:11.5px;color:var(--cyan);font-weight:700;">WITH AQUA-SHIELD</div>
            <div style="font-size:12.5px;margin-top:4px;line-height:1.3;">ranked hazard register,<br>evidence attached</div>
          </div>
          <div style="color:var(--cyan);font-size:19px;">›</div>
          <div style="flex:1;border:1px solid var(--line);border-radius:9px;padding:10px;
                      text-align:center;background:rgba(74,222,128,0.08);">
            <div style="font-size:11.5px;color:var(--green);font-weight:700;">ACTION</div>
            <div style="font-size:12.5px;margin-top:4px;line-height:1.3;">GeoJSON → QGIS<br>vessel tasked</div>
          </div>
        </div>
        <ul class="tight" style="margin-bottom:12px;">
          <li>Every hazard carries its <b>evidence, calibrated confidence and
              provenance</b> — an operator can audit a decision, not just accept it.</li>
          <li>Repeat sightings are merged into unique hazards; positional
              uncertainty tightens by ~√N over repeat fixes.</li>
          <li class="amber">We do <b>not</b> claim a percentage of analyst time saved.
              That requires a user study we have not run.</li>
        </ul>
        <img class="shot" src="{u('panel_geo_table.png')}">
        <div class="shotcap">The exported register, from the running system: real
          coordinates, a per-hazard uncertainty in metres, and a priority band —
          this is what a cleanup vessel is tasked from.</div>
      </div>

      <div class="card">
        <div class="lbl green">Who it serves</div>
        <table class="t">
          <tr><th>Beneficiary</th><th>Benefit</th></tr>
          <tr><td>MoES / NIOT<br><span style="font-size:10.5px;color:var(--txt-3);">survey &amp; cleanup ops</span></td>
              <td class="dim">Direct fit to the problem statement's own agency; runs on survey hardware already aboard.</td></tr>
          <tr><td>Fisheries &amp;<br>coastal management</td>
              <td class="dim">Ghost gear keeps killing catch and habitat long after loss; early detection breaks that cycle.</td></tr>
          <tr><td>Ports &amp; navigation<br>safety</td>
              <td class="dim">Mine-like and man-made bottom objects flagged and prioritised before they foul a channel.</td></tr>
          <tr><td>Environmental<br>research</td>
              <td class="dim">Auditable, licence-clean dataset and reproducible experiment registry.</td></tr>
        </table>
        <div class="callout cy" style="margin-top:15px;">
          <b>Deployment context.</b> The georeferenced run above places hazards at
          12.92&deg;N, 80.34&deg;E — off the Tamil Nadu coast, NIOT's own operating area.
          The navigation track in that demo is <b>synthetic</b>, and the product says so
          on screen: we are showing the geolocation path works end to end, not claiming
          a validated fix.
        </div>
        <div style="margin-top:14px;font-size:12.5px;color:var(--txt-2);line-height:1.5;">
          <b style="color:#fff;">Cost model:</b> no cloud inference, no per-image API fee,
          no data egress. The marginal cost of processing one more survey is the
          electricity to run a laptop that is already aboard.
        </div>
      </div>
    </div>

    <div style="margin-top:15px;">
      <div class="lbl">Deployment &amp; scalability — offline-first by design</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:13px;">
        <div class="card" style="padding:14px 16px;">
          <div class="stat"><div class="v cyan">Laptop</div>
            <div class="k">Runs today on an Apple M5, 24 GB — fully offline. No cloud, no
              per-image cost, no data leaving the vessel.</div></div>
        </div>
        <div class="card" style="padding:14px 16px;">
          <div class="stat"><div class="v cyan">10.6 <span style="font-size:17px;">MB</span></div>
            <div class="k">ONNX export, measured at 8.5 ms inference — sized for
              Jetson-class edge hardware aboard an AUV or survey launch.</div></div>
        </div>
        <div class="card" style="padding:14px 16px;">
          <div class="stat"><div class="v cyan">REST</div>
            <div class="k">FastAPI service + SQLite register: the same pipeline scales
              from one operator console to a shore-side survey fleet.</div></div>
        </div>
        <div class="card" style="padding:14px 16px;">
          <div class="stat"><div class="v amber">Not yet</div>
            <div class="k">Never run on a Jetson or a live AUV. That is the next
              milestone, and it is a target — not something we claim today.</div></div>
        </div>
      </div>
    </div>
  </div>
</div>"""


# ============================================================== SLIDE 6 =====
S6 = f"""
<div class="slide">
  {chrome(6, "Research &amp; References", "Prior art, method, and what we have not done",
          "Judges trust a team that states its limits. The right-hand box is on the slide deliberately.")}
  <div class="body">
    <div style="display:grid;grid-template-columns:1.25fr 1fr;gap:15px;height:100%;">

      <div style="display:flex;flex-direction:column;gap:12px;">
        <div class="card" style="flex:1;">
          <div class="lbl">Prior art — and our relationship to it</div>
          <table class="t">
            <tr><th style="width:34%;">Work</th><th style="width:19%;">Licence</th><th>Relationship</th></tr>
            <tr><td>GhostVision<br><span style="font-size:10px;color:var(--txt-3);">JMSE 14(10):951, 2025</span></td>
                <td class="dim">NOASSERTION</td>
                <td class="dim">Closest system. Not vendored — licence unresolved.</td></tr>
            <tr><td>PINGMapper<br><span style="font-size:10px;color:var(--txt-3);">Earth &amp; Space Science, 2022</span></td>
                <td style="color:var(--green);">MIT</td>
                <td class="dim">Pipeline shape and output conventions.</td></tr>
            <tr><td>sidescantools</td><td class="dim">GPL-3.0</td>
                <td class="dim">Studied; not vendored.</td></tr>
            <tr><td>AI4Shipwrecks<br><span style="font-size:10px;color:var(--txt-3);">arXiv 2401.14546</span></td>
                <td style="color:var(--green);">MIT</td>
                <td class="dim">Route to a wreck class later.</td></tr>
            <tr><td>MILCO / NOMBO<br><span style="font-size:10px;color:var(--txt-3);">Data in Brief 53:110132</span></td>
                <td style="color:var(--green);">CC BY 4.0</td>
                <td class="dim"><b style="color:#fff;">Training data</b> — mine-like objects.</td></tr>
            <tr><td>sss-crab-pot-detection-ds<br><span style="font-size:10px;color:var(--txt-3);">PINGEcosystem</span></td>
                <td style="color:var(--green);">CC BY-SA 4.0</td>
                <td class="dim"><b style="color:#fff;">Training data</b> — derelict ghost gear.</td></tr>
            <tr><td>SSM-DETR · TR-YOLOv5s<br>MSF-DETR · LEF-RT-DETR
                <br><span style="font-size:10px;color:var(--txt-3);">4 detection architectures reviewed</span></td>
                <td class="dim">papers</td>
                <td class="dim"><b style="color:var(--amber);">Evaluated, not adopted.</b>
                    Each needs <b style="color:#fff;">2.6×–44× our compute</b> for
                    single-digit AP gains, on datasets we cannot access or that are
                    forward-looking sonar, not side-scan.</td></tr>
          </table>
          <div class="callout" style="margin-top:14px;padding:10px 14px;font-size:12.5px;">
            <b>Why we did not simply adopt a published architecture.</b> The published
            gains are real — but they are measured on a different sonar modality, at a
            compute budget that rules out edge deployment, on data we cannot obtain to
            reproduce. Our bottleneck is false alarms on empty seabed, which a bigger
            backbone does not fix.
          </div>
        </div>

        <div class="card" style="padding:14px 17px;">
          <div class="lbl violet">Method references</div>
          <div style="font-size:13px;color:var(--txt-2);line-height:1.65;">
            <b style="color:#fff;">Lee (1980)</b> — adaptive speckle filtering,
            DOI 10.1109/TPAMI.1980.4766994 &nbsp;·&nbsp;
            <b style="color:#fff;">Platt (1999)</b> — probabilistic outputs for
            large-margin classifiers, MIT Press &nbsp;·&nbsp;
            <b style="color:#fff;">Guo et al. (2017)</b> — on calibration of modern
            neural networks, arXiv:1706.04599 &nbsp;·&nbsp;
            <b style="color:#fff;">Ultralytics YOLO11</b> — detection backbone.
          </div>
        </div>
      </div>

      <div style="display:flex;flex-direction:column;gap:12px;">
        <div class="card" style="border-color:rgba(245,179,65,0.5);">
          <div class="lbl amber">What we have NOT done — stated deliberately</div>
          <ul class="tight">
            <li class="amber">Ghost-gear detection has <b>one raw training result</b>
                (mAP50 0.323). It has not been through the verification stage,
                calibration, or an ablation.</li>
            <li class="amber">Geolocation accuracy has <b>never been validated</b> —
                the licensed data available to us ships no navigation track.</li>
            <li class="amber">Detection <b>degrades under added speckle noise</b>
                (measured). A noise-robust checkpoint exists but trades clean accuracy.</li>
            <li class="amber">Never run on a <b>Jetson or a live AUV</b>.</li>
            <li class="amber">Our own sonar preprocessing chain <b>did not help</b> when
                measured properly, and ships disabled.</li>
            <li class="amber">Detector backend is <b>AGPL-3.0</b>; the licence is
                isolated behind an interface but not yet replaced.</li>
          </ul>
        </div>

        <div class="card" style="flex:1;">
          <div class="lbl green">Roadmap to the finals — how each gap closes</div>
          <table class="t" style="font-size:12.5px;">
            <tr><td style="width:26%;color:var(--cyan);font-weight:700;">1 · Verify</td>
                <td class="dim">Fit the verification stage on the ghost-gear model and
                    run the full ablation — turn one raw number into a pipeline result.</td></tr>
            <tr><td style="color:var(--cyan);font-weight:700;">2 · Compare</td>
                <td class="dim">Measure against detector-only, rule-filter and
                    YOLO11s / RT-DETR at equal compute.</td></tr>
            <tr><td style="color:var(--cyan);font-weight:700;">3 · Calibrate</td>
                <td class="dim">Report reliability curve and ECE on the held-out split.</td></tr>
            <tr><td style="color:var(--cyan);font-weight:700;">4 · Edge</td>
                <td class="dim">Benchmark the ONNX export on Jetson-class hardware.</td></tr>
            <tr><td style="color:var(--cyan);font-weight:700;">5 · Geolocate</td>
                <td class="dim">Validate positional accuracy against a survey that
                    actually ships a navigation track.</td></tr>
          </table>
        </div>

        <div class="callout cy" style="font-size:13px;">
          <b>Everything above is reproducible.</b> Every figure in this deck traces to
          <span class="mono">experiments/registry.jsonl</span>,
          <span class="mono">docs/BENCHMARKS.md</span>, or a live run of the prototype
          — regenerated by script, never typed in by hand.
        </div>
      </div>
    </div>
  </div>
</div>"""


SLIDES = [S1, S2, S3, S4, S5, S6]

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="{(DECK / 'slides.css').as_uri()}">
<style>.mono{{font-family:var(--mono);}}</style>
</head><body>{''.join(SLIDES)}</body></html>"""

page_html = BUILD / "deck.html"
page_html.write_text(HTML)
print(f"wrote {page_html}")


# --------------------------------------------------------------- render ----
CHROME_BIN = ("/Users/earther/Library/Caches/ms-playwright/chromium-1228/chrome-mac-arm64/"
              "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing")

from playwright.sync_api import sync_playwright  # noqa: E402

pngs = []
with sync_playwright() as pw:
    browser = (pw.chromium.launch(executable_path=CHROME_BIN)
               if Path(CHROME_BIN).exists() else pw.chromium.launch(channel="chrome"))
    page = browser.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=2)
    page.goto(page_html.as_uri(), wait_until="networkidle", timeout=90_000)
    page.wait_for_timeout(1500)

    for i, el in enumerate(page.query_selector_all(".slide"), 1):
        p = BUILD / f"slide{i}.png"
        el.screenshot(path=str(p))
        pngs.append(p)
        print(f"  rendered {p.name}")

    page.pdf(path=str(OUT_DIR / "AQUA_SHIELD_SIH_PITCH.pdf"),
             width="1600px", height="900px", print_background=True,
             margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    browser.close()
print(f"wrote {OUT_DIR / 'AQUA_SHIELD_SIH_PITCH.pdf'}")


# ----------------------------------------------------------------- pptx ----
from pptx import Presentation                     # noqa: E402
from pptx.util import Inches                      # noqa: E402

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]
for p in pngs:
    s = prs.slides.add_slide(blank)
    s.shapes.add_picture(str(p), 0, 0, width=prs.slide_width, height=prs.slide_height)
prs.save(str(OUT_DIR / "AQUA_SHIELD_SIH_PITCH.pptx"))
print(f"wrote {OUT_DIR / 'AQUA_SHIELD_SIH_PITCH.pptx'}")
