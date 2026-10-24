"""
Step 5: Drought detection from Water Balance & NDVI anomalies.
Uses Step 4 functions to build collections, then computes per-pixel z-scores
and classifies drought severity per year.

Method:
  1. z-score of water balance and of NDVI, per pixel, across all years
  2. combined stress index = -(wb_z + ndvi_z) / 2   (positive = drier / less green)
  3. the combined index is re-standardised (divided by its own multi-year std),
     because the average of two z-scores has std < 1
  4. classes on the standardised index:
       0 none (<0.5)  1 mild (0.5-1)  2 moderate (1-1.5)
       3 severe (1.5-2)  4 extreme (>=2)
"""

import sys
import ee
from step2_boundaries import get_mykolaiv_boundaries
from step3_farmland_mask import get_cropland_250m
from step4_indicators import get_water_balance, get_ndvi

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ID = ""
YEARS = list(range(2000, 2026))   # 2000 ... 2025 inclusive
MONTHS = list(range(4, 10))       # April ... September (6 months)
WAR_YEARS = {2022, 2023, 2024, 2025}   # NDVI may reflect abandoned/mined fields

CLASS_NAMES = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Extreme"}


def compute_zscore(collection, band_name):
    """
    Per-pixel z-score across the years of the collection.
    Returns the same images with an extra band '<band_name>_zscore'.
    """
    band = collection.select(band_name)
    mean_img = band.mean()
    std_img = band.reduce(ee.Reducer.stdDev())
    std_img = std_img.updateMask(std_img.gt(0))   # avoid division by zero

    def zscore_one(img):
        z = (img.select(band_name)
             .subtract(mean_img)
             .divide(std_img)
             .rename(f"{band_name}_zscore"))
        return img.addBands(z).copyProperties(img, ["year"])

    return collection.map(zscore_one)


def classify_drought(combined_z):
    """
    Turn the standardised combined index into classes 0-4.
    Pixels that are masked in the index (non-farmland, no data) stay masked.
    """
    drought = (ee.Image(0)
               .where(combined_z.gte(0.5), 1)
               .where(combined_z.gte(1.0), 2)
               .where(combined_z.gte(1.5), 3)
               .where(combined_z.gte(2.0), 4)
               .updateMask(combined_z.mask())
               .rename("drought_class"))
    return drought


def get_drought_collection(years, months, crop_mask):
    """
    One image per year with bands:
      water_balance, water_balance_zscore, ndvi, ndvi_zscore,
      combined_z, drought_class
    """
    n = len(years)
    wb_zlist = compute_zscore(get_water_balance(years, months, crop_mask),
                              "water_balance").toList(n)
    ndvi_zlist = compute_zscore(get_ndvi(years, months, crop_mask),
                                "ndvi").toList(n)

    wb_imgs = [ee.Image(wb_zlist.get(i)) for i in range(n)]
    ndvi_imgs = [ee.Image(ndvi_zlist.get(i)) for i in range(n)]

    # Combined stress index per year (positive = drought-like)
    raw = [wb.select("water_balance_zscore")
             .add(nd.select("ndvi_zscore"))
             .multiply(-0.5)
             .rename("combined_raw")
           for wb, nd in zip(wb_imgs, ndvi_imgs)]

    raw_col = ee.ImageCollection(raw)
    mean_c = raw_col.mean()
    std_c = raw_col.reduce(ee.Reducer.stdDev())
    std_c = std_c.updateMask(std_c.gt(0))

    images = []
    for year, wb, nd, r in zip(years, wb_imgs, ndvi_imgs, raw):
        combined_z = r.subtract(mean_c).divide(std_c).rename("combined_z")
        drought = classify_drought(combined_z)
        img = (ee.Image.cat([
                    wb.select("water_balance"),
                    wb.select("water_balance_zscore"),
                    nd.select("ndvi"),
                    nd.select("ndvi_zscore"),
                    combined_z,
                    drought])
               .set("year", year)
               .set("system:time_start", ee.Date.fromYMD(year, months[0], 1).millis()))
        images.append(img)

    return ee.ImageCollection(images)


def zonal_drought_stats(drought_col, region, modis_proj):
    """
    Area (km2) per drought class per year, counted on the MODIS 250 m grid
    (same grid as Step 4).
    Returns {year: {class_int: km2}}.
    """
    def one_year(img):
        year = ee.Number(img.get("year"))
        drought = img.select("drought_class")
        # Count only farmland pixels that have a valid class
        area_img = (ee.Image.pixelArea().divide(1e6)
                    .updateMask(drought.mask())
                    .rename("area_km2"))

        # Group reducer needs group field as LAST input (area first, then class)
        stats = area_img.addBands(drought).reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName="drought_class"),
            geometry=region,
            crs=modis_proj,
            scale=modis_proj.nominalScale(),
            maxPixels=1e13,
            tileScale=6,
        )

        groups = ee.List(stats.get("groups"))
        keys = groups.map(
            lambda g: ee.Number(ee.Dictionary(g).get("drought_class")).format("%d"))
        vals = groups.map(lambda g: ee.Dictionary(g).get("sum"))
        return ee.Feature(None, {"year": year,
                                 "stats": ee.Dictionary.fromLists(keys, vals)})

    feats = ee.FeatureCollection(drought_col.map(one_year)).getInfo()["features"]
    return {
        int(f["properties"]["year"]):
            {int(k): float(v) for k, v in f["properties"]["stats"].items()}
        for f in feats
    }


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5: Drought Detection from WB & NDVI Anomalies")
    print("=" * 60)

    ee.Initialize(project=PROJECT_ID)

    oblast, districts = get_mykolaiv_boundaries()
    region = oblast.geometry()
    share250, mask250, modis_proj = get_cropland_250m()

    print(f"Years: {YEARS[0]}-{YEARS[-1]} ({len(YEARS)} years)")
    print(f"Season: Months {MONTHS[0]}-{MONTHS[-1]} (Apr-Sep)")

    print("\nBuilding drought assessment collection...")
    drought_col = get_drought_collection(YEARS, MONTHS, mask250)
    print(f"  Collection size: {drought_col.size().getInfo()} images")

    print("\nComputing zonal drought statistics (area per class)...")
    stats = zonal_drought_stats(drought_col, region, modis_proj)

    print(f"\n{'Year':<7}{'None':>8}{'Mild':>8}{'Moderate':>10}"
          f"{'Severe':>8}{'Extreme':>9}{'Total km2':>11}")
    print("-" * 61)
    for y in YEARS:
        s = stats.get(y, {})
        areas = [s.get(c, 0.0) for c in range(5)]
        total = sum(areas)
        mark = "*" if y in WAR_YEARS else ""
        print(f"{str(y) + mark:<7}{areas[0]:>8.0f}{areas[1]:>8.0f}{areas[2]:>10.0f}"
              f"{areas[3]:>8.0f}{areas[4]:>9.0f}{total:>11.0f}")
    print("* = war year: NDVI may be affected by abandoned fields, mines, "
          "front-line activity")

    print("\nMajor drought years (>20% farmland in Moderate+):")
    for y in YEARS:
        s = stats.get(y, {})
        total = sum(s.get(c, 0.0) for c in range(5))
        drought_km2 = sum(s.get(c, 0.0) for c in (2, 3, 4))
        if total > 0 and drought_km2 / total > 0.2:
            flag = "  (war year)" if y in WAR_YEARS else ""
            print(f"  {y}: {drought_km2:.0f} km2 ({100 * drought_km2 / total:.1f}%){flag}")

    print("\n[OK] Step 5 complete: Drought detection ready")
    print("  - Per-pixel z-scores for WB & NDVI")
    print("  - Standardised combined index, 5 severity classes")
    print("  - Zonal statistics by year on the MODIS 250 m grid")
    print("  - Collection available for Step 6 (visualization/export)")