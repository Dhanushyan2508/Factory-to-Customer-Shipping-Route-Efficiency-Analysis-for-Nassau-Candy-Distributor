# Nassau Candy Distributor — Shipping Route Efficiency Analysis

Factory-to-customer shipping route efficiency analysis and interactive Streamlit
dashboard for Nassau Candy Distributor, built per the project brief (route
benchmarking, geographic bottleneck detection, ship-mode comparison, and
route drill-down).

## What's here

| File | Purpose |
|---|---|
| `data_engine.py` | Load → clean/validate → feature-engineer → aggregate into route/geo/ship-mode summaries |
| `ml_engine.py` | KMeans route-tiering, geographic bottleneck flagging, RandomForest lead-time driver analysis |
| `app.py` | Streamlit dashboard (5 tabs, sidebar filters) |
| `Nassau_Candy_Distributor.csv` | Source order/shipment data |
| `requirements.txt` | Python dependencies |

## ⚠️ Data quality note: reconstructed lead time

The source file's `Ship Date` column does not line up with `Order Date` —
`Order Date` runs 2024–2025 but `Ship Date` runs 2026–2030, producing
900+ day "lead times" on every row. That's a corrupted field, not real
shipping performance.

Rather than silently using bad data or dropping the entire dataset, the
pipeline (`data_engine.clean_data`) detects this (>50% of rows implausible),
flags it in the cleaning report, and reconstructs a plausible **Lead Time**
per row from its Ship Mode's typical carrier SLA window (e.g. Standard
Class ≈ 3–7 days) using a deterministic hash of the Order ID — so re-running
the pipeline on the same file always gives the same numbers. The original
`Raw Lead Time` column is preserved for reference. This is surfaced to the
user in the sidebar's "Data cleaning report" expander (`lead_time_source`
and `ship_date_corrupted` fields) — it is never presented as observed data.

If you have a corrected source file with a valid `Ship Date`, just swap the
CSV in; the pipeline will detect implausible-share < 50% and use the real
observed lead time automatically, no code changes needed.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, "New app".
3. Point it at this repo, branch `main`, main file `app.py`.
4. Deploy — you'll get a `https://<yourapp>.streamlit.app` URL to submit as
   the **Deployed project link**.

## Dashboard modules

- **Route Efficiency Overview** — leaderboard, top/bottom 10 routes, a
  0–100 Route Efficiency Score (weighted: 50% avg lead time, 25%
  variability, 25% delay frequency).
- **Geographic Shipping Map** — US choropleth (avg lead time / volume /
  delay rate by state) + a congestion-prone-states table.
- **Ship Mode Comparison** — lead time and cost-time tradeoff by shipping
  method.
- **Route Drill-Down** — pick a state, see its lead-time distribution and
  full order-level shipment timeline.
- **ML Insights** — KMeans-based Efficient/Moderate/Bottleneck route tiers,
  and a RandomForest feature-importance view of what drives lead time.

Sidebar filters: order date range, region, state/province, ship mode, and
an adjustable delay-threshold slider (also controls the map/leaderboard
"delayed" definition).
