#!/usr/bin/env python3
"""Capture real, high-resolution screenshots of the running AQUA-SHIELD dashboard.

These are REAL screenshots of the real prototype processing real held-out survey
frames -- not mockups, not renders. Used in the SIH deck so a judge sees the
actual system, and regenerable so they never drift from the code.

Prereq: the dashboard must already be running, e.g.
    .venv/bin/streamlit run dashboard/app.py --server.port 8521 --server.headless true
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8521"
OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "prototype"
OUT.mkdir(parents=True, exist_ok=True)

VIEWPORT = {"width": 1500, "height": 1320}
SCALE = 2  # 2x device pixels => print-quality raster for slides

CHROME = ("/Users/earther/Library/Caches/ms-playwright/chromium-1228/chrome-mac-arm64/"
          "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing")


def shot(page, name):
    p = OUT / f"{name}.png"
    page.screenshot(path=str(p))
    print(f"  wrote {p.name}")


def click_tab(page, label):
    page.get_by_role("tab", name=label, exact=False).first.click(timeout=15_000)
    page.wait_for_timeout(2500)


def run_scenario(page, scenario, prefix, tabs):
    """Select a demo scenario in the sidebar, process it, capture the given tabs."""
    if scenario:
        print(f"scenario: {scenario}")
        box = page.get_by_label("Scenario", exact=False).first
        box.click()
        page.wait_for_timeout(600)
        page.get_by_text(scenario, exact=True).last.click()
        page.wait_for_timeout(1200)

    page.get_by_role("button", name="Process survey").click()
    page.wait_for_timeout(32_000)

    for label, name in tabs:
        try:
            if label:
                click_tab(page, label)
            shot(page, f"{prefix}{name}")
        except Exception as e:
            print(f"  skipped {label}: {type(e).__name__}")


with sync_playwright() as pw:
    browser = (pw.chromium.launch(executable_path=CHROME)
               if Path(CHROME).exists() else pw.chromium.launch(channel="chrome"))
    page = browser.new_page(viewport=VIEWPORT, device_scale_factor=SCALE)
    page.goto(URL, wait_until="networkidle", timeout=90_000)
    page.wait_for_timeout(3000)

    # default scenario (01_clear_targets): detection + verification evidence
    run_scenario(page, None, "", [
        (None, "01_detections"),
        ("Hazard register", "03_register"),
        ("Evidence & QC", "04_evidence"),
        ("Export", "05_export"),
        ("Provenance", "06_provenance"),
    ])

    # georeferenced scenario: this is the one that can actually place hazards on a map
    run_scenario(page, "04_georeferenced", "geo_", [
        ("Map", "02_map"),
        ("Hazard register", "03_register"),
    ])

    browser.close()
print("done")
