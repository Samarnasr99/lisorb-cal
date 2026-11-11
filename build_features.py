"""Turn the app's raw inputs (database column names, X__*) into the 84 model inputs with the same code as the training pipeline."""
import numpy as np, pandas as pd
import li_features as L

RAW_COLS = ['X__material_family', 'X__material_class', 'X__material_name_reported', 'X__adsorbent_form', 'X__binder_or_support', 'X__active_content_percent', 'X__particle_size_um', 'X__BET_m2_g',
            'X__pore_size_nm', 'X__pore_volume_cm3_g', 'X__dopant', 'X__dopant_content', 'X__synthesis_route', 'X__calcination_temp_C', 'X__acid_treatment_agent', 'X__acid_treatment_conc_M', 'X__mechanism_class',
            'X__feed_matrix_complexity', 'X__brine_type', 'X__brine_source', 'X__Li_mg_L', 'X__Na_mg_L', 'X__K_mg_L', 'X__Mg_mg_L', 'X__Ca_mg_L', 'X__Cl_mg_L', 'X__SO4_mg_L', 'X__B_mg_L', 'X__Sr_mg_L', 'X__TDS_mg_L',
            'X__other_ions_note', 'X__initial_pH', 'X__temperature_C', 'X__contact_time_min', 'X__adsorbent_dose_g_L', 'X__solution_volume_mL', 'X__agitation_rpm', 'X__experiment_type', 'X__flow_rate_mL_min', 'X__bed_volume_mL',
            'X__cycle_number', 'X__desorption_agent', 'X__desorption_conc_M', 'X__desorption_temp_C', 'X__desorption_time_min', 'meta__extractor_note']

def raw_frame(records):
    """records: list of dicts keyed by X__ column (missing -> NaN)."""
    df = pd.DataFrame(records);
    for c in RAW_COLS:
        if c not in df: df[c] = np.nan
    df = df[RAW_COLS].copy()
    Li, Na, K, Mg, Ca = [pd.to_numeric(df['X__' + e + '_mg_L'], errors='coerce') for e in ['Li', 'Na', 'K', 'Mg', 'Ca']]
    pos = Li > 0
    D = {'D__Mg_Li_ratio': Mg / Li, 'D__Na_Li_ratio': Na / Li, 'D__K_Li_ratio': K / Li, 'D__Ca_Li_ratio': Ca / Li, 'D__divalent_Li_ratio': (Mg + Ca) / Li,
         'D__cation_competition_index': (Na + K + Mg + Ca) / Li, 'D__log_Mg_Li': np.log10((Mg / Li).where(Mg > 0)), 'D__sum_cations_mg_L': Li + Na + K + Mg + Ca}
    for c, v in D.items(): df[c] = v.where(pos)
    df['meta__extractor_note'] = df['meta__extractor_note'].fillna('')
    return df

def features(records, feature_list, cats, vocab):
    F = L.full_features(raw_frame(records))
    for c in feature_list:
        if c not in F: F[c] = np.nan
    X = F[feature_list].copy()
    for c in cats:
        X[c] = pd.Categorical(X[c].astype(str), categories=vocab[c]).codes.astype(int)
    return X

class ADPrep:
    """Applicability-domain preprocessing: log1p on skewed columns, median imputation, standardisation (fitted on training rows)."""
    def __init__(self, cols, logc, imp, sc): self.cols, self.logc, self.imp, self.sc = cols, logc, imp, sc
    def transform(self, X):
        Q = X[self.cols].copy()
        for c in self.logc: Q[c] = np.log1p(Q[c].clip(lower=0))
        return self.sc.transform(self.imp.transform(Q))

class StackedModel:
    """Picklable deployment form of the stacked, bagged ensemble: prediction = Σ_w w · mean(bag predictions)."""
    def __init__(self, members):  # members: list of (weight, [fitted estimators])
        self.members = members
    def predict(self, X):
        return sum(w * np.mean([m.predict(X) for m in ms], axis=0) for w, ms in self.members)

class NF:
    """NaN -> sentinel wrapper used for the scikit-learn forests inside the ensemble (same as in the training pipeline)."""
    def __init__(self, m): self.m = m
    def fit(self, X, y): self.m.fit(X.fillna(-999), y); return self
    def predict(self, X): return self.m.predict(X.fillna(-999))

# the training pipeline pickled NF under __main__; make the saved bundles loadable from any entry point
import sys as _sys
_sys.modules['__main__'].NF = NF
