#!/usr/bin/env python3
# Copyright 2025 Nicolas Cravino
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Step 5: Benchmark — Before vs After Comparison

Generates a side-by-side HTML report with quality scores and live
rendered previews showing baseline vs adapted model output for each
test prompt.

All test prompts produce HTML landing pages. Scoring checks for:
  - Valid HTML structure (DOCTYPE, html, head, body)
  - Semantic sections (header, nav, main, section, footer)
  - CSS styling (style tag with rules)
  - Content sections (hero, services/features, about, contact)
  - Visual richness (colors, fonts, spacing)

Usage:
    python 05_benchmark.py
    python 05_benchmark.py --run-dir runs/my-test
"""

import argparse
import json
import re
import html
from pathlib import Path


# ---------------------------------------------------------------------------
# HTML Landing Page quality scoring
# ---------------------------------------------------------------------------

def measure_html_quality(raw: str) -> dict:
    """Comprehensive quality metrics for generated HTML landing pages."""
    metrics = {
        "chars": len(raw),
        "has_doctype": False,
        "has_html_tag": False,
        "has_head": False,
        "has_body": False,
        "has_style_tag": False,
        "css_rule_count": 0,
        "has_header": False,
        "has_footer": False,
        "has_nav": False,
        "has_main_or_sections": False,
        "section_count": 0,
        "has_hero_content": False,
        "has_services_content": False,
        "has_about_content": False,
        "has_contact_info": False,
        "has_colors": False,
        "has_font_family": False,
        "has_padding_margin": False,
        "total_html_tags": 0,
        "is_renderable": False,
    }

    low = raw.lower()

    # Basic HTML structure
    metrics["has_doctype"] = "<!doctype html>" in low
    metrics["has_html_tag"] = "<html" in low and "</html>" in low
    metrics["has_head"] = "<head" in low and "</head>" in low
    metrics["has_body"] = "<body" in low and "</body>" in low

    # CSS
    style_match = re.search(r'<style[^>]*>(.*?)</style>', raw, re.DOTALL | re.IGNORECASE)
    if style_match:
        metrics["has_style_tag"] = True
        css = style_match.group(1)
        metrics["css_rule_count"] = len(re.findall(r'\{[^}]+\}', css))
        metrics["has_colors"] = bool(re.search(r'(color\s*:|background|#[0-9a-fA-F]{3,8}|rgb)', css))
        metrics["has_font_family"] = "font-family" in css.lower()
        metrics["has_padding_margin"] = bool(re.search(r'(padding|margin)\s*:', css))

    # Semantic HTML elements
    metrics["has_header"] = "<header" in low
    metrics["has_footer"] = "<footer" in low
    metrics["has_nav"] = "<nav" in low
    section_count = low.count("<section")
    metrics["section_count"] = section_count
    metrics["has_main_or_sections"] = "<main" in low or section_count >= 2

    # Content detection (look for typical landing page patterns)
    # Hero: h1 tag or large heading near top
    metrics["has_hero_content"] = bool(re.search(r'<h1[^>]*>.*?</h1>', raw, re.DOTALL | re.IGNORECASE))

    # Services/features: look for multiple h2/h3 or list items or repeated patterns
    h2_count = len(re.findall(r'<h[23][^>]*>', raw, re.IGNORECASE))
    li_count = len(re.findall(r'<li[^>]*>', raw, re.IGNORECASE))
    metrics["has_services_content"] = h2_count >= 2 or li_count >= 3

    # About section
    metrics["has_about_content"] = bool(re.search(r'(about|our story|who we are|since \d{4})', low))

    # Contact info
    metrics["has_contact_info"] = bool(re.search(r'(phone|email|address|contact|\d{3}[-.)]\s*\d{3})', low))

    # Total HTML tags (measure of richness)
    metrics["total_html_tags"] = len(re.findall(r'<[a-zA-Z][^>]*>', raw))

    # Is it even renderable HTML?
    metrics["is_renderable"] = (
        metrics["has_html_tag"]
        and metrics["has_body"]
        and metrics["total_html_tags"] >= 10
    )

    return metrics


def quality_score(metrics: dict) -> int:
    """0-100 quality score for HTML landing pages."""
    score = 0

    # === Structure (30 points) ===
    if metrics["has_doctype"]:
        score += 5
    if metrics["has_html_tag"]:
        score += 5
    if metrics["has_head"]:
        score += 5
    if metrics["has_body"]:
        score += 5
    if metrics["has_header"]:
        score += 5
    if metrics["has_footer"]:
        score += 5

    # === Styling (25 points) ===
    if metrics["has_style_tag"]:
        score += 8
    if metrics["css_rule_count"] >= 5:
        score += 5
    if metrics["css_rule_count"] >= 15:
        score += 4
    if metrics["has_colors"]:
        score += 4
    if metrics["has_font_family"]:
        score += 2
    if metrics["has_padding_margin"]:
        score += 2

    # === Content (30 points) ===
    if metrics["has_hero_content"]:
        score += 8
    if metrics["has_services_content"]:
        score += 8
    if metrics["has_about_content"]:
        score += 7
    if metrics["has_contact_info"]:
        score += 7

    # === Richness (15 points) ===
    if metrics["has_main_or_sections"]:
        score += 5
    if metrics["section_count"] >= 3:
        score += 3
    if metrics["total_html_tags"] >= 30:
        score += 3
    if metrics["total_html_tags"] >= 60:
        score += 2
    if metrics["has_nav"]:
        score += 2

    return min(score, 100)


def _make_default_paths(run_dir: str) -> dict:
    """Build a paths dict for standalone usage (not via pipeline)."""
    root = Path(run_dir)
    return {
        "root":            str(root),
        "outputs_before":  str(root / "outputs" / "before"),
        "outputs_after":   str(root / "outputs" / "after"),
        "outputs_teacher": str(root / "outputs" / "teacher"),
        "comparison":      str(root / "outputs" / "comparison.html"),
    }


def generate_comparison(paths: dict | None = None, run_dir: str = ".") -> bool:
    """Generate comparison HTML. Returns True on success."""
    if paths is None:
        paths = _make_default_paths(run_dir)

    run_path = Path(paths["root"])

    with open("scenarios/prompts.json") as f:
        prompts = json.load(f)

    test_prompts = prompts["test_prompts"]
    before_dir = Path(paths["outputs_before"])
    after_dir = Path(paths["outputs_after"])
    teacher_dir = Path(paths.get("outputs_teacher", paths["root"] + "/outputs/teacher"))

    if not before_dir.exists() or not after_dir.exists():
        print(f"✗ Need both {before_dir}/ and {after_dir}/")
        return False

    has_teacher = teacher_dir.exists() and any(teacher_dir.glob("test_*.*"))

    # Determine student/teacher model names from run_config if available
    teacher_name = "Claude teacher"
    student_name = "Student"
    run_config_path = run_path / "run_config.json"
    if run_config_path.exists():
        try:
            with open(run_config_path) as f:
                cfg = json.load(f)
            teacher_name = cfg.get("teacher_model", teacher_name)
            student_name = cfg.get("student_model", student_name)
            if "/" in student_name:
                student_name = student_name.split("/")[-1]
        except Exception:
            pass

    def _find_output(directory: Path, idx: int) -> str:
        """Find output file, trying multiple extensions."""
        for ext in [".html", ".json", ".md", ".txt"]:
            f = directory / f"test_{idx}{ext}"
            if f.exists():
                return f.read_text()
        return "(not generated)"

    # Extract a short name for each test (business name)
    def _extract_biz_name(prompt: str) -> str:
        m = re.search(r'called\s+"([^"]+)"', prompt)
        if m:
            return m.group(1)
        m = re.search(r'for a\s+(\w+(?:\s+\w+)?)', prompt)
        return m.group(1).title() if m else "Website"

    tabs = []
    for i, prompt in enumerate(test_prompts):
        idx = i + 1
        before_raw = _find_output(before_dir, idx)
        after_raw = _find_output(after_dir, idx)
        teacher_raw = _find_output(teacher_dir, idx) if has_teacher else ""

        before_metrics = measure_html_quality(before_raw)
        after_metrics = measure_html_quality(after_raw)
        teacher_metrics = measure_html_quality(teacher_raw)
        before_score = quality_score(before_metrics)
        after_score = quality_score(after_metrics)
        teacher_score = quality_score(teacher_metrics)

        biz_name = _extract_biz_name(prompt)

        def detail_badges(m):
            badges = []
            if m["is_renderable"]:
                badges.append(("Renderable", "green"))
            else:
                badges.append(("Broken HTML", "red"))
            if m["has_style_tag"]:
                badges.append((f"CSS: {m['css_rule_count']} rules", "blue"))
            else:
                badges.append(("No CSS", "red"))
            if m["has_hero_content"]:
                badges.append(("Hero ✓", "purple"))
            if m["has_services_content"]:
                badges.append(("Services ✓", "purple"))
            if m["has_contact_info"]:
                badges.append(("Contact ✓", "gray"))
            badges.append((f"{m['total_html_tags']} tags", "gray"))
            return badges

        tabs.append({
            "index": idx,
            "prompt": prompt,
            "biz_name": biz_name,
            "before_raw": before_raw,
            "after_raw": after_raw,
            "teacher_raw": teacher_raw,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "teacher_metrics": teacher_metrics,
            "before_score": before_score,
            "after_score": after_score,
            "teacher_score": teacher_score,
            "before_badges": detail_badges(before_metrics),
            "after_badges": detail_badges(after_metrics),
            "teacher_badges": detail_badges(teacher_metrics),
        })

    # Build the HTML report
    tab_buttons = ""
    tab_panels = ""

    for tab in tabs:
        active = "active" if tab["index"] == 1 else ""
        tab_buttons += (
            f'<button class="tab-btn {active}" onclick="showTab({tab["index"]})" '
            f'id="tab-btn-{tab["index"]}">{html.escape(tab["biz_name"])}</button>\n'
        )

        display = "flex" if tab["index"] == 1 else "none"
        improvement = tab["after_score"] - tab["before_score"]
        imp_color = "#22c55e" if improvement > 0 else "#ef4444" if improvement < 0 else "#888"

        def badges_html(badges):
            out = ""
            colors = {"green": "#22c55e", "red": "#ef4444", "blue": "#3b82f6", "purple": "#a855f7", "gray": "#6b7280"}
            for text, color in badges:
                c = colors.get(color, "#6b7280")
                out += f'<span class="badge" style="background:{c}20;color:{c};border:1px solid {c}40">{html.escape(text)}</span> '
            return out

        # Encode HTML for srcdoc (escape for HTML attribute)
        def srcdoc_escape(raw_html: str) -> str:
            # For srcdoc we need to escape quotes and ampersands
            return raw_html.replace("&", "&amp;").replace('"', "&quot;")

        # Determine view mode (default: rendered preview)
        before_srcdoc = srcdoc_escape(tab["before_raw"])
        after_srcdoc = srcdoc_escape(tab["after_raw"])
        teacher_srcdoc = srcdoc_escape(tab["teacher_raw"])

        before_code = html.escape(tab["before_raw"])
        after_code = html.escape(tab["after_raw"])
        teacher_code = html.escape(tab["teacher_raw"])

        teacher_panel = ""
        if has_teacher:
            teacher_panel = f'''
    <div class="panel">
      <div class="panel-header teacher">
        <span>{html.escape(teacher_name)}</span>
        <span class="score">Score: {tab["teacher_score"]}/100</span>
      </div>
      <div class="badge-bar">{badges_html(tab["teacher_badges"])}</div>
      <div class="panel-content" id="panel-{tab["index"]}-teacher">
        <iframe class="preview-frame" srcdoc="{teacher_srcdoc}" sandbox="allow-same-origin"></iframe>
        <pre class="code-view" style="display:none"><code>{teacher_code}</code></pre>
      </div>
    </div>'''

        metrics_line = (
            f'Score: {tab["before_score"]} → {tab["after_score"]} '
            f'(<span style="color:{imp_color}">{"+" if improvement >= 0 else ""}{improvement}</span>)'
        )
        if has_teacher:
            metrics_line += f' &nbsp;|&nbsp; Teacher: {tab["teacher_score"]}'

        tab_panels += f'''
<div class="tab-panel" id="tab-panel-{tab["index"]}" style="display: {display}">
  <div class="prompt-bar">
    <div class="prompt-left">
      <strong>Test {tab["index"]}:</strong> {html.escape(tab["biz_name"])}
      <div class="metrics">{metrics_line} &nbsp;|&nbsp; Size: {tab["before_metrics"]["chars"]:,} → {tab["after_metrics"]["chars"]:,} chars</div>
    </div>
    <div class="view-toggle">
      <button class="toggle-btn active" onclick="setView({tab["index"]}, 'preview')">Preview</button>
      <button class="toggle-btn" onclick="setView({tab["index"]}, 'code')">Code</button>
    </div>
  </div>
  <div class="comparison">
    <div class="panel">
      <div class="panel-header before">
        <span>{html.escape(student_name)} — Base</span>
        <span class="score">Score: {tab["before_score"]}/100</span>
      </div>
      <div class="badge-bar">{badges_html(tab["before_badges"])}</div>
      <div class="panel-content" id="panel-{tab["index"]}-before">
        <iframe class="preview-frame" srcdoc="{before_srcdoc}" sandbox="allow-same-origin"></iframe>
        <pre class="code-view" style="display:none"><code>{before_code}</code></pre>
      </div>
    </div>
    <div class="panel">
      <div class="panel-header after">
        <span>{html.escape(student_name)} — Distilled</span>
        <span class="score">Score: {tab["after_score"]}/100</span>
      </div>
      <div class="badge-bar">{badges_html(tab["after_badges"])}</div>
      <div class="panel-content" id="panel-{tab["index"]}-after">
        <iframe class="preview-frame" srcdoc="{after_srcdoc}" sandbox="allow-same-origin"></iframe>
        <pre class="code-view" style="display:none"><code>{after_code}</code></pre>
      </div>
    </div>{teacher_panel}
  </div>
</div>
'''

    run_id = run_path.name if str(run_path) != "." else "local"

    # Compute summary stats
    avg_before = sum(t["before_score"] for t in tabs) / len(tabs)
    avg_after = sum(t["after_score"] for t in tabs) / len(tabs)
    avg_teacher = sum(t["teacher_score"] for t in tabs) / len(tabs)
    renderable_before = sum(1 for t in tabs if t["before_metrics"]["is_renderable"])
    renderable_after = sum(1 for t in tabs if t["after_metrics"]["is_renderable"])
    renderable_teacher = sum(1 for t in tabs if t["teacher_metrics"].get("is_renderable", False))
    css_before = sum(1 for t in tabs if t["before_metrics"]["has_style_tag"])
    css_after = sum(1 for t in tabs if t["after_metrics"]["has_style_tag"])
    css_teacher = sum(1 for t in tabs if t["teacher_metrics"].get("has_style_tag", False))

    comparison_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Distillation Demo — {run_id}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f0f0f; color: #e0e0e0; }}
.header {{
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  padding: 24px 32px;
  border-bottom: 1px solid #333;
}}
.header h1 {{ font-size: 1.5rem; color: #fff; margin-bottom: 4px; }}
.header p {{ color: #888; font-size: 0.9rem; }}
.summary-bar {{
  display: flex; gap: 24px; padding: 12px 32px;
  background: #151520; border-bottom: 1px solid #333;
  font-size: 0.85rem; flex-wrap: wrap;
}}
.summary-stat {{ color: #aaa; }}
.summary-stat strong {{ color: #fff; }}
.tabs {{
  display: flex; gap: 4px; padding: 12px 32px;
  background: #1a1a1a; border-bottom: 1px solid #333;
}}
.tab-btn {{
  padding: 8px 20px; background: #2a2a2a; border: 1px solid #444;
  border-radius: 6px; color: #aaa; cursor: pointer; font-size: 0.9rem; transition: all 0.2s;
}}
.tab-btn:hover {{ background: #333; color: #fff; }}
.tab-btn.active {{ background: #3b82f6; border-color: #3b82f6; color: #fff; }}
.prompt-bar {{
  padding: 12px 32px; background: #1e1e1e;
  border-bottom: 1px solid #333; font-size: 0.85rem; color: #ccc;
  display: flex; justify-content: space-between; align-items: center;
}}
.prompt-left {{ flex: 1; }}
.prompt-bar .metrics {{ margin-top: 4px; font-size: 0.8rem; color: #888; }}
.view-toggle {{ display: flex; gap: 2px; }}
.toggle-btn {{
  padding: 5px 14px; background: #2a2a2a; border: 1px solid #444;
  color: #aaa; cursor: pointer; font-size: 0.8rem; transition: all 0.2s;
}}
.toggle-btn:first-child {{ border-radius: 4px 0 0 4px; }}
.toggle-btn:last-child {{ border-radius: 0 4px 4px 0; }}
.toggle-btn.active {{ background: #3b82f6; border-color: #3b82f6; color: #fff; }}
.comparison {{ display: flex; height: calc(100vh - 260px); gap: 2px; background: #333; }}
.panel {{ flex: 1; display: flex; flex-direction: column; background: #1a1a2e; overflow: hidden; }}
.panel-header {{
  padding: 10px 16px; font-size: 0.85rem; font-weight: 600;
  display: flex; justify-content: space-between; align-items: center;
  flex-shrink: 0;
}}
.panel-header.before {{ background: #2d1b1b; color: #fca5a5; border-bottom: 2px solid #ef4444; }}
.panel-header.after {{ background: #1b2d1b; color: #86efac; border-bottom: 2px solid #22c55e; }}
.panel-header.teacher {{ background: #1b1b2d; color: #93c5fd; border-bottom: 2px solid #3b82f6; }}
.panel-header .score {{ font-weight: 400; font-size: 0.8rem; }}
.badge-bar {{ padding: 6px 12px; display: flex; gap: 6px; flex-wrap: wrap; flex-shrink: 0; background: #12121e; }}
.badge {{
  font-size: 0.7rem; padding: 2px 8px; border-radius: 4px; font-weight: 500;
  white-space: nowrap;
}}
.panel-content {{
  flex: 1; overflow: hidden; position: relative;
}}
.preview-frame {{
  width: 100%; height: 100%; border: none; background: #fff;
}}
.code-view {{
  width: 100%; height: 100%; overflow: auto; margin: 0;
  padding: 12px 16px; background: #0d0d1a;
  font-size: 0.75rem; line-height: 1.5;
}}
.code-view code {{
  font-family: "SF Mono", "Fira Code", "Consolas", monospace;
  color: #c9d1d9; white-space: pre-wrap; word-break: break-all;
}}
.tab-panel {{ flex-direction: column; }}
</style>
</head>
<body>
<div class="header">
  <h1>Distillation Demo — Landing Page Generation</h1>
  <p>Run: {run_id} | {html.escape(student_name)} — Base vs Distilled vs {html.escape(teacher_name)} Teacher</p>
</div>
<div class="summary-bar">
  <span class="summary-stat">Base Avg: <strong>{avg_before:.0f}</strong></span>
  <span class="summary-stat">Distilled Avg: <strong>{avg_after:.0f}</strong> ({"+" if avg_after >= avg_before else ""}{avg_after - avg_before:+.0f})</span>
  {"<span class='summary-stat'>Teacher Avg: <strong>" + f"{avg_teacher:.0f}" + "</strong></span>" if has_teacher else ""}
  <span class="summary-stat">Renderable: <strong>{renderable_before} → {renderable_after}{f" → {renderable_teacher}" if has_teacher else ""} / {len(tabs)}</strong></span>
  <span class="summary-stat">Has CSS: <strong>{css_before} → {css_after}{f" → {css_teacher}" if has_teacher else ""} / {len(tabs)}</strong></span>
  <span class="summary-stat">Tests: <strong>{len(tabs)}</strong></span>
</div>
<div class="tabs">
{tab_buttons}
</div>
{tab_panels}
<script>
function showTab(n) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.style.display = 'none');
  document.getElementById('tab-btn-' + n).classList.add('active');
  document.getElementById('tab-panel-' + n).style.display = 'flex';
}}

function setView(tabIdx, mode) {{
  const panel = document.getElementById('tab-panel-' + tabIdx);
  if (!panel) return;

  // Update toggle buttons
  const toggles = panel.querySelectorAll('.toggle-btn');
  toggles.forEach(b => b.classList.remove('active'));
  if (mode === 'preview') toggles[0].classList.add('active');
  else toggles[1].classList.add('active');

  // Toggle iframes and code views
  panel.querySelectorAll('.preview-frame').forEach(f => f.style.display = mode === 'preview' ? 'block' : 'none');
  panel.querySelectorAll('.code-view').forEach(c => c.style.display = mode === 'code' ? 'block' : 'none');
}}
</script>
</body>
</html>'''

    out_file = Path(paths["comparison"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(comparison_html)
    print(f"✓ Comparison page saved to {out_file}")

    print("\n" + "=" * 70)
    print("  Quality Score Summary")
    print("=" * 70)

    for tab in tabs:
        imp = tab["after_score"] - tab["before_score"]
        arrow = "↑" if imp > 0 else "↓" if imp < 0 else "→"
        check_b = "✓" if tab["before_metrics"]["is_renderable"] else "✗"
        check_a = "✓" if tab["after_metrics"]["is_renderable"] else "✗"
        check_t = "✓" if tab["teacher_metrics"].get("is_renderable") else "✗"
        name = tab["biz_name"][:20].ljust(20)
        if has_teacher:
            print(f"  Test {tab['index']:>2d} [{name}]:  {tab['before_score']:>3d}  {tab['after_score']:>3d}  {tab['teacher_score']:>3d}   {arrow}{imp:+3d}   {check_b} → {check_a} → {check_t}")
        else:
            print(f"  Test {tab['index']:>2d} [{name}]:  {tab['before_score']:>3d}  {tab['after_score']:>3d}   {arrow}{imp:+3d}   {check_b} → {check_a}")

    print()
    if has_teacher:
        print(f"  Average:    {avg_before:.0f}  →  {avg_after:.0f}  →  {avg_teacher:.0f}   (base→dist: {avg_after - avg_before:+.0f})")
    else:
        print(f"  Average: {avg_before:.0f} → {avg_after:.0f}  (Δ {avg_after - avg_before:+.0f})")
    print(f"  Renderable: {renderable_before}/{len(tabs)} → {renderable_after}/{len(tabs)}" +
          (f" → {renderable_teacher}/{len(tabs)}" if has_teacher else ""))
    print(f"  Has CSS:    {css_before}/{len(tabs)} → {css_after}/{len(tabs)}" +
          (f" → {css_teacher}/{len(tabs)}" if has_teacher else ""))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate side-by-side comparison")
    parser.add_argument("--run-dir", default=".", help="Run directory")
    args = parser.parse_args()

    success = generate_comparison(run_dir=args.run_dir)  # standalone: uses _make_default_paths
    if not success:
        exit(1)
