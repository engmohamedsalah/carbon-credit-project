#!/usr/bin/env python3
"""
One real end-to-end run of the fixed forest-cover / carbon pipeline.

Run this on a machine WITH torch (Python 3.12 — torch has no 3.14 wheels). It takes a
real Sentinel-2 GeoTIFF, runs the fixed process_single_image, and prints the reported
area + tCO2e so you can confirm they match the real scene. It also reads the GeoTIFF's
actual pixel size from its geotransform to catch a subtle bug the code review didn't:
calculate_carbon_impact hardcodes pixel_area_m2 = 100 (i.e. assumes 10m pixels) — if your
GeoTIFF is at a different resolution, the area is still off by (real_res/10m)^2.

Usage:
    python ml/smoke_test_pipeline.py path/to/sentinel2_scene.tif

What to check in the output:
    1. "reported total area" should match the scene's real hectares (not a fixed ~41 ha).
    2. "pixel size" should be ~10 m; if not, the hardcoded 100 m2/pixel is wrong for this file.
    3. Eyeball the forest mask stats — at F1 0.49 on a non-Amazon scene the mask may be poor,
       which is exactly why a human-review step belongs before any report goes out.
"""
import os
import sys
import argparse

# Make `ml` importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("geotiff", help="Path to a Sentinel-2 GeoTIFF scene")
    args = ap.parse_args()

    if not os.path.exists(args.geotiff):
        sys.exit(f"File not found: {args.geotiff}")

    # --- read the raster's real geometry (no torch needed) ---
    try:
        import rasterio
        with rasterio.open(args.geotiff) as src:
            bands, H, W = src.count, src.height, src.width
            t = src.transform
            px_x, px_y = abs(t.a), abs(t.e)  # pixel size in CRS units
            crs = src.crs
        is_metric = bool(crs and crs.is_projected)
        print("=" * 64)
        print(f"Scene: {args.geotiff}")
        print(f"  bands={bands}  size={H}x{W} px  CRS={crs}  projected={is_metric}")
        if is_metric:
            px_area = px_x * px_y
            real_ha = H * W * px_area / 10_000
            print(f"  pixel size ~ {px_x:.1f} x {px_y:.1f} m  -> pixel area {px_area:.0f} m2")
            print(f"  REAL scene ground area (from geotransform): {real_ha:,.1f} ha")
            if abs(px_x - 10) > 1:
                print(f"  ⚠️  pixel size is not ~10 m — the pipeline hardcodes 100 m2/pixel, "
                      f"so its area will be off by ~{(px_x/10)**2:.2f}x. Fix pixel_area_m2.")
        else:
            print("  ⚠️  CRS is geographic (degrees) — cannot compute ground area directly; "
                  "reproject to UTM for a meaningful check.")
    except Exception as e:
        print(f"Could not read raster geometry: {e}")

    # --- run the real pipeline (needs torch) ---
    try:
        from ml.inference.production_inference import CarbonCreditVerificationPipeline
    except Exception as e:
        sys.exit(f"\nCannot import the pipeline (torch/deps missing?): {e}\n"
                 f"Run this on a Python 3.12 environment with the ML deps installed.")

    print("\nRunning fixed process_single_image ...")
    pipeline = CarbonCreditVerificationPipeline(device="cpu")
    result = pipeline.process_single_image(args.geotiff, output_name="smoke_test")
    if not result:
        sys.exit("Pipeline returned no result (check logs above).")

    ci = result["carbon_impact"]
    fp = result["forest_prediction"]
    print("=" * 64)
    print("RESULT")
    print(f"  reported total area : {ci['total_area_hectares']:,.1f} ha")
    print(f"  forest area         : {ci['forest_area_hectares']:,.1f} ha "
          f"({ci['forest_coverage_percent']:.1f}% cover)")
    print(f"  carbon              : {ci['total_co2e_tonnes']:,.0f} tCO2e  (biome={ci.get('biome')})")
    print(f"  forest mask         : mean_prob={fp['mean_probability']:.3f} "
          f"min={fp['min_probability']:.3f} max={fp['max_probability']:.3f}")
    print("=" * 64)

    # --- sanity checks ---
    ok = True
    if abs(ci["total_area_hectares"] - 40.96) < 0.5 and (H, W) != (64, 64):
        print("❌ area is stuck at ~41 ha but the scene is larger — the crop bug is NOT fixed here.")
        ok = False
    else:
        print("✅ area scales with the real scene (not capped at ~41 ha).")
    print("\nNow eyeball it: does 'reported total area' match the scene's real hectares above,\n"
          "and is the forest mask plausible? If the mask looks wrong, that's the F1~0.49 limit —\n"
          "route it through human review before any report is issued.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
