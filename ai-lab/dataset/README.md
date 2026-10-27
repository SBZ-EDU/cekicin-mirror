---
license: cc-by-4.0
language: [tr, en]
pretty_name: Çekicin AI Knowledge Base
tags: [roadside-assistance, car-damage, repair-cost, openstreetmap, emissions, turkey]
size_categories: [n<1K]
configs:
  - config_name: damage_classes
    data_files: damage_classes.csv
  - config_name: parts_costs
    data_files: parts_costs.csv
  - config_name: cardd_costs
    data_files: cardd_costs.csv
  - config_name: decision_rules
    data_files: decision_rules.csv
  - config_name: segments
    data_files: segments.csv
  - config_name: settings
    data_files: settings.csv
  - config_name: roadside_services
    data_files: roadside_services.csv
  - config_name: emission_factors
    data_files: emission_factors.csv
  - config_name: faq
    data_files: faq.csv
---

# Çekicin AI Knowledge Base

The data layer behind the **[Çekicin AI Lab](https://huggingface.co/spaces/sosa123454321/cekicin-ai-lab)**.
The Space reads these CSV files at runtime, so editing a file here changes the Space with no redeploy.

| Table | Used by | What it holds |
|---|---|---|
| `damage_classes` | Damage detection | The 14 classes of [`cekicin-damage-yolo11n-onnx`](https://huggingface.co/sosa123454321/cekicin-damage-yolo11n-onnx) → car part (the 6-class academic model is [`cekicin-cardd-yolov8s-seg-onnx`](https://huggingface.co/sosa123454321/cekicin-cardd-yolov8s-seg-onnx)) + damage type, TR/EN labels |
| `cardd_costs` | Academic CarDD model | Action + cost by CarDD damage type (dent, scratch, crack, glass, lamp, tyre) and size |
| `decision_rules` | Repair / replace | Action by damage type and damage size (box area ÷ image area) |
| `parts_costs` | Cost estimate | Part price range, repair / paint / replace labour hours per part |
| `segments` | Cost estimate | Price multiplier per vehicle segment |
| `settings` | Cost estimate | Labour rate, paint cost per hour |
| `roadside_services` | Nearby search | Roadside needs → OpenStreetMap tags (free Overpass API) |
| `emission_factors` | Impact calculator | UK DESNZ 2025 conversion factors, with sources |
| `faq` | Voice / text assistant | Question–answer pairs; rows with the same `topic` are TR/EN translations |

## Prices (updated 2026-10-23, sourced)

Placeholders were replaced with **sourced 2026 prices for Türkiye, Germany, the UK and the USA**:

| File | What |
|---|---|
| `cost-model.json` | Full model used by the site (model-test country selector): labour / paint rates, parts min–max, repair rules, every source URL |
| `cost_countries.csv` | Per country: currency, VAT, reference car, body / paint labour rate range, paint material per hour |
| `parts_prices_by_country.csv` | 15 parts × 4 countries, min–max price, `estimate=1` where no live source was found |
| `ato_2026_paint_tr.csv` | Türkiye: ATO 2026 maximum paint price per panel (excl. VAT) |
| `parts_costs.csv`, `settings.csv` | Same columns as before (TRY) so the API keeps working; values now from the TR sources |

Reference cars: TR Fiat Egea · DE VW Golf 7 · GB Ford Fiesta · US Toyota Corolla 2020. Main sources: ATO (Ankara Ticaret Odası) 2026
tariff, armut.com, fiatparcacim / otocikma / parcario / n11 (TR); DEKRA DRS 2026, autodoc.de (DE); AM-online June 2026, autodoc.co.uk,
shop.ford.co.uk (GB); NABR June 2026, RockAuto, Toyota OEM (US). Not covered yet: UAE, Iran (no reliable public sources found).
It is still a **rough pre-estimate from one photo**, not a quote.

## Cost formula used by the Space

```
part      = (min+max)/2 × segment multiplier           (replace only)
labour    = hours(action) × labour_rate_try
paint     = paint_h × paint_material_try_per_h          (repair_paint / replace of painted panels)
range     = total × 0.8 … total × 1.25
```

Emission factors are copied from the UK DESNZ 2025 GHG conversion factors (see the `source`/`url` columns).
The tow-truck value uses a rigid diesel HGV (7.5–17 t, fully laden) as a proxy.
