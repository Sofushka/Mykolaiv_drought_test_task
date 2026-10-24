"""
Step 3: Farmland (cropland) mask from ESA WorldCover 2021 (v200), class 40.
Builds two versions:
  - 10 m   : for measuring farmland area accurately
  - 250 m  : on the MODIS grid, for the vegetation analysis in later steps
"""

import ee
from step2_boundaries import get_mykolaiv_boundaries  # reuse Step 2, don't copy it

PROJECT_ID = ""
CROPLAND_CLASS = 40
MIN_CROP_SHARE = 0.3   # keep a 250 m cell if >= 30% of it is cropland


def get_cropland_10m():
    """1 = cropland, 0 = anything else (kept as 0/1, not masked)."""
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
    return worldcover.select("Map").eq(CROPLAND_CLASS).rename("cropland")


def get_cropland_250m():
    """
    Cropland on the MODIS NDVI grid.
    Returns (share, mask): share = fraction of each cell that is cropland (0-1),
    mask = 1 where share >= MIN_CROP_SHARE.
    """
    modis_proj = ee.ImageCollection("MODIS/061/MOD13Q1").first().projection()
    share = (
        get_cropland_10m()
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs=modis_proj)
        .rename("crop_share")
    )
    mask = share.gte(MIN_CROP_SHARE).rename("crop_mask")
    return share, mask, modis_proj


def area_km2(image01, region, scale):
    """Total area (km2) where a 0/1 image equals 1, inside the region."""
    pixel_area = ee.Image.pixelArea().divide(1e6).rename("area_km2")
    result = pixel_area.updateMask(image01.eq(1)).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        tileScale=6,
    )
    return result.get("area_km2").getInfo()


if __name__ == "__main__":
    ee.Initialize(project=PROJECT_ID)

    oblast, districts = get_mykolaiv_boundaries()
    region = oblast.geometry()

    crop10 = get_cropland_10m()
    share250, mask250, modis_proj = get_cropland_250m()

    oblast_km2 = region.area(maxError=100).divide(1e6).getInfo()
    crop10_km2 = area_km2(crop10, region, scale=30)
    modis_scale = modis_proj.nominalScale().getInfo()
    crop250_km2 = area_km2(mask250, region, scale=modis_scale)

    print(f"\nOblast area:                  {oblast_km2:,.0f} km2")
    print(f"Cropland (10 m map):          {crop10_km2:,.0f} km2 "
          f"({100 * crop10_km2 / oblast_km2:.0f}% of oblast)")
    print(f"Cropland mask (250 m, >=30%): {crop250_km2:,.0f} km2 "
          f"(ratio to 10 m: {crop250_km2 / crop10_km2:.2f})")

    