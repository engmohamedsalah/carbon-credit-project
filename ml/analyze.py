"""End-to-end carbon analysis: project boundary -> Sentinel-2 -> ML -> PDF report.

Chains the free-STAC acquisition (ml/acquisition/sentinel2.py), the fixed forest-cover
pipeline (ml/inference/production_inference.py), and a ReportLab PDF into one call. This
is the "full loop" for a project boundary. It runs wherever torch + rasterio + reportlab
are available (a real ML runtime / locally) — NOT the Vercel serverless backend.

The report states its own limitations honestly: the forest model is a research prototype
(F1~0.49 in-distribution, weaker and conservative out-of-region — see
docs/ml-forest-model-investigation.md), so every figure needs human verification.
"""
import io
import os
from datetime import datetime

import numpy as np
import rasterio
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from ml.acquisition.sentinel2 import fetch_sentinel2_stack, geometry_bbox
from ml.inference.production_inference import CarbonCreditVerificationPipeline

_PIPELINE = None


def _pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = CarbonCreditVerificationPipeline(device="cpu")
    return _PIPELINE


def analyze_boundary(geometry, out_pdf, project_name="Untitled Project",
                     start_date="2023-06-01", end_date="2023-09-30", max_cloud=10):
    """Boundary (GeoJSON, WGS84) -> imagery -> carbon analysis -> PDF report.

    Returns {pdf, scene, carbon, forest_prediction}. Raises on fetch/inference failure.
    """
    work_tif = os.path.splitext(out_pdf)[0] + "_scene.tif"
    scene = fetch_sentinel2_stack(geometry, work_tif, start_date, end_date, max_cloud)
    result = _pipeline().process_single_image(work_tif, output_name="analysis")
    if not result:
        raise RuntimeError("pipeline returned no result (see logs)")
    _render_pdf(out_pdf, project_name, geometry, scene,
                result["carbon_impact"], result["forest_prediction"], work_tif)
    return {"pdf": out_pdf, "scene": scene,
            "carbon": result["carbon_impact"], "forest_prediction": result["forest_prediction"]}


def _rgb_thumbnail(scene_path, max_px=520):
    """True-color RGB preview (B04,B03,B02 -> bands 4,3,2) as PNG bytes, 2-98% stretched."""
    from PIL import Image as PILImage
    with rasterio.open(scene_path) as s:
        r, g, b = s.read(4), s.read(3), s.read(2)

    def stretch(x):
        x = x.astype(np.float32)
        vals = x[x > 0]
        lo, hi = (np.percentile(vals, [2, 98]) if vals.size else (0.0, 1.0))
        return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

    rgb = (np.dstack([stretch(r), stretch(g), stretch(b)]) * 255).astype(np.uint8)
    img = PILImage.fromarray(rgb)
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return buf


def _kv_table(rows):
    t = Table(rows, colWidths=[6 * cm, 9.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f5e9")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8e6c9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _render_pdf(out_pdf, project_name, geometry, scene, ci, fp, scene_path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_pdf, pagesize=A4, title="Forest Carbon Analysis",
                            author="Carbon Credit Verification (prototype)")
    story = []
    story.append(Paragraph("Forest Carbon Analysis",
                           ParagraphStyle("t", parent=styles["Title"],
                                          textColor=colors.HexColor("#1b5e20"))))
    story.append(Paragraph(project_name, styles["Heading2"]))
    story.append(Spacer(1, 0.25 * cm))

    lon0, lat0, lon1, lat1 = geometry_bbox(geometry)
    story.append(_kv_table([
        ["Boundary (WGS84)", f"{lon0:.4f}, {lat0:.4f}  →  {lon1:.4f}, {lat1:.4f}"],
        ["Imagery", f"Sentinel-2 L2A  {scene['scene_id']}"],
        ["Acquired", scene["date"]],
        ["Cloud cover", f"{scene['cloud']:.1f}%"],
        ["Analyzed area", f"{ci['total_area_hectares']:,.0f} ha"],
    ]))
    story.append(Spacer(1, 0.3 * cm))
    try:
        story.append(Image(_rgb_thumbnail(scene_path), width=8 * cm, height=8 * cm))
        story.append(Paragraph("True-color preview (Sentinel-2 B4/B3/B2)",
                               ParagraphStyle("cap", parent=styles["Normal"],
                                              fontSize=7, textColor=colors.grey)))
    except Exception:
        pass
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("Carbon Estimate", styles["Heading2"]))
    story.append(_kv_table([
        ["Forest coverage", f"{ci['forest_coverage_percent']:.1f}%"],
        ["Forest area", f"{ci['forest_area_hectares']:,.0f} ha"],
        ["Carbon stock", f"{ci['total_co2e_tonnes']:,.0f} tCO2e"],
        ["Carbon density", f"{ci['carbon_density_tco2e_per_ha']:.0f} tCO2e/ha ({ci['biome']})"],
        ["Mean forest probability", f"{fp['mean_probability']:.2f}"],
    ]))
    story.append(Spacer(1, 0.45 * cm))

    story.append(Paragraph(
        "<b>Experimental — not for compliance-grade crediting.</b> The forest-cover model "
        "is a research prototype (F1≈0.49 in-distribution; weaker and conservative "
        "out-of-region — it tends to UNDER-estimate forest). Every figure requires "
        "independent human verification before any use. Ground area is derived from the "
        "image geotransform; carbon uses IPCC biome densities × 3.67 (CO₂:C).",
        ParagraphStyle("w", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor("#b71c1c"))))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"Generated {datetime.now():%Y-%m-%d %H:%M} — Carbon Credit Verification (prototype)",
        ParagraphStyle("f", parent=styles["Normal"], fontSize=7, textColor=colors.grey)))
    doc.build(story)


if __name__ == "__main__":
    forest_aoi = {"type": "Polygon", "coordinates": [[
        [-59.67, -3.30], [-59.63, -3.30], [-59.63, -3.25], [-59.67, -3.25], [-59.67, -3.30]]]}
    meta = analyze_boundary(forest_aoi, "carbon_report_selfcheck.pdf",
                            project_name="Demo Amazon Plot")
    assert os.path.exists(meta["pdf"]) and os.path.getsize(meta["pdf"]) > 2000, "PDF not written"
    c = meta["carbon"]
    print(f"OK  pdf={meta['pdf']} ({os.path.getsize(meta['pdf'])} bytes)  "
          f"coverage={c['forest_coverage_percent']:.1f}%  {c['total_co2e_tonnes']:,.0f} tCO2e")
