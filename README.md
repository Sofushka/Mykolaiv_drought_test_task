# Drought in Mykolaiv Oblast, 2000–2025

Finds the drought years in Mykolaiv Oblast (Ukraine) from open satellite and climate data, and shows how much farmland each one affected. Built with Python and Google Earth Engine.

## Result

Drought hit **7 of 26 years** (a year counts if more than 20% of farmland was Moderate-or-worse):

| Year | Farmland in Moderate or worse |
|------|------------------------------|
| 2007 | 87% |
| 2024* | 83% |
| 2012 | 77% |
| 2025* | 66% |
| 2017 | 41% |
| 2018 | 28% |
| 2020 | 23% |

\*War year: satellite greenness may also reflect abandoned or mined fields. The water-balance data alone also marks 2024 and 2025 as very dry.

All four districts show drought in 6–8 of the 26 years, so the data does not rank them clearly.

## Method

1. **Area:** oblast and 4 districts from FAO GAUL 2025.
2. **Farmland only:** ESA WorldCover 2021 (cropland), on the MODIS 250 m grid, cells with at least 30% cropland.
3. **Two indicators per year, April–September:**
   - water balance = rain minus potential evaporation (ERA5-Land), in mm;
   - NDVI, a satellite greenness measure (MODIS MOD13Q1), cloudy data removed.
4. **How unusual was the year:** per-pixel z-score of each indicator against 2000–2025.
5. **Combine and classify:** average the two z-scores (dry and less green = drought), standardise, then classify: None (<0.5), Mild (0.5–1), Moderate (1–1.5), Severe (1.5–2), Extreme (≥2).
6. **Count:** farmland area (km²) per class, per year, for the oblast and each district.

## Deliverables

| File | What it is |
|------|------------|
| `results.pptx` | 4-slide presentation for farmers |
| `methodology_mykolaiv_drought.pdf` | Methodology description |
| `prompts_log_mykolaiv_drought.pdf` | Text of the AI prompts used |

## Code

| File | What it does |
|------|--------------|
| `step1_earth_engine_setup.py` | Connects to Earth Engine, tests the datasets |
| `step2_boundaries.py` | Oblast and district boundaries |
| `step3_farmland_mask.py` | Farmland mask (10 m and 250 m) |
| `step4_indicators.py` | Yearly water balance and NDVI |
| `step5_drought_detection.py` | Z-scores, drought classes, area per class |
| `step6_visualization.py` | Time-series chart, maps, district table, HTML report in `outputs/` |
| `step7_validation.py` | Data checks, report in `outputs/step7_validation_report.txt` |

## Run

```bash
pip install earthengine-api matplotlib
earthengine authenticate
```

Set `PROJECT_ID` (top of each script) to your own Google Cloud project that is registered for Earth Engine, then run the steps in order:

```bash
python step1_earth_engine_setup.py
python step5_drought_detection.py   # steps 2-4 are imported by it
python step6_visualization.py
python step7_validation.py
```

## Limitations

- Classes are **relative to each place's own 2000–2025 history**; a place is Moderate-or-worse in roughly one year in six by construction. Look for years when a large share is hit at once.
- Water balance uses potential evaporation, which is high in ERA5-Land. Read it as a difference between years, not as a real water deficit.
- ERA5-Land is about 11 km resolution; NDVI is 250 m and carries most of the local detail.
- NDVI also depends on the crop and field use, not only on drought. The 2021 land cover is applied to all years.
- Not validated against yields or ground soil-moisture data.

## Data sources

[FAO GAUL 2025](https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2025_level1) · [ESA WorldCover v200](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200) · [MODIS MOD13Q1 v061](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1) · [ERA5-Land monthly](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR) · CHIRPS (rain cross-check in step 7)
