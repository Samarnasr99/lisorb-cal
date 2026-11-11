# LiSorb-Cal — calibrated property predictor for lithium-selective sorbents

Streamlit application accompanying the manuscript *"Transferability of literature-trained machine-learning models for lithium-selective sorbents: publication-grouped nested validation and one-point calibration"*.

It predicts 19 properties of lithium-selective sorbents (spinel Mn-oxide and Ti-oxide ion sieves, Li/Al layered double hydroxides) from material, feed and operating descriptors. The models are the deployed stacked ensembles refitted on the whole database (7,946 experiments from 450 publications). Every reliability statement in the app comes from the nested, publication-grouped evaluation of the paper, i.e. from publications the model had never seen.

## Pages

| Page | What it shows |
|---|---|
| Predict | All 12 main (or 19) properties with 90% empirical intervals and a reliability label based on the R² on unseen publications |
| Calibrate with your data | Enter one or more measured values; the app estimates your study's offset with Eq. (1) of the paper and corrects the predictions; a measured capacity also calibrates kinetic qe and Langmuir qmax |
| What-if curves | Prediction along one varied condition (C₀, dose, contact time, pH, temperature) within the literature range |
| Why this prediction | Largest input contributions (LightGBM component, or exact SHAP of the whole ensemble) |
| Applicability domain | Distance to the five nearest training experiments, leverage, and the input placed on the Williams plot of the literature rows |
| Literature neighbours | The ten most similar published experiments, with publication and DOI |
| Batch | Predict (and calibrate by `material_id`) a whole CSV file |
| How good is the model? | Parity plots, random split versus unseen publications, with your own measurement before and after calibration |
| About the models | Rows, publications, R² (random and unseen), offset share and λ for every property |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11–3.13. The model bundles in `models/` (about 300 MB) are loaded once and cached.

## Deploy (free)

**Streamlit Community Cloud.** Push this folder to a public GitHub repository (keep `app.py`, `build_features.py`, `li_features.py`, `requirements.txt` and `models/` at the root; the largest model file is about 76 MB, below GitHub's 100 MB per-file limit). At https://share.streamlit.io choose *New app*, select the repository and `app.py`, set Python 3.12 under *Advanced settings*, and deploy. Put the resulting URL in the Data availability statement of the paper.

**Hugging Face Spaces** is an alternative: create a Space with the Streamlit SDK and upload the same files (Git LFS for `models/`).

## Files

* `models/<target>.joblib` — deployed ensemble, feature list, vocabulary, interval half-widths, λ and μ̂ for Eq. (1), applicability-domain reference
* `models/diagnostics/<target>.npz` — nested and random-split predictions and Williams-plot coordinates used by the parity and applicability pages
* `models/deploy_manifest.json` — SHA-256 hashes and stacking weights of the deployed models
* `models/cross_calibration.json` — gains for calibration across properties
