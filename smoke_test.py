"""Headless smoke test: python smoke_test.py  (runs one prediction per target without the UI)."""
import json, joblib, numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import build_features as BF
from build_features import NF, StackedModel, ADPrep
IDX = json.load(open(Path(__file__).parent / 'models/index.json'))
rec = {'X__material_family': 'Mn_LMO', 'X__material_name_reported': 'HMO', 'X__adsorbent_form': 'powder', 'X__Li_mg_L': 100, 'X__initial_pH': 10, 'X__temperature_C': 25, 'X__contact_time_min': 1440, 'X__adsorbent_dose_g_L': 1.0, 'X__experiment_type': 'batch', 'X__cycle_number': 1}
for t in IDX['targets']:
    B = joblib.load(Path(__file__).parent / f'models/{t}.joblib'); X = BF.features([rec], B['features'], B['cats'], B['vocab'])
    p = B['model'].predict(X)[0]; d = B['ad']['nn'].kneighbors(B['ad']['prep'].transform(X), 5)[0].mean(1)[0]
    print(f"{IDX['labels'][t]:28s} prediction {p:9.3f}  AD distance {d:5.2f} (limit {B['ad']['threshold']:.2f})  ±{B['meta']['pi_halfwidth']['0']:.3f}")
