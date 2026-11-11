"""LiSorb-Cal: calibrated property predictor for lithium-selective sorbents (Streamlit).
Companion tool of the paper "Transferability of literature-trained machine-learning models for lithium-selective sorbents:
publication-grouped nested validation and one-point calibration". Models are the deployed stacked ensembles refitted on all rows of the
database (450 publications); every reliability statement shown comes from the nested, publication-grouped evaluation of the paper."""
import json, sys, numpy as np, pandas as pd, streamlit as st, joblib
from pathlib import Path
import plotly.graph_objects as go
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE)); import build_features as BF
from build_features import NF, StackedModel, ADPrep  # noqa: F401  (classes referenced by the pickled bundles)
sys.modules['__main__'].NF = NF
MODELS = HERE / 'models'
st.set_page_config(page_title='LiSorb-Cal', page_icon='🧪', layout='wide')
st.markdown("""<style>
div[data-testid="stMetricValue"] {font-size: 1.35rem;}
.badge {display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.78rem;font-weight:600;color:white}
.good {background:#1a7f5a} .fair {background:#c98a00} .poor {background:#8a8a8a}
.card {border:1px solid #e3e3e3;border-radius:10px;padding:10px 12px;margin-bottom:8px}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ data
@st.cache_resource
def index(): return json.load(open(MODELS / 'index.json'))
@st.cache_resource
def bundle(t): return joblib.load(MODELS / f'{t}.joblib')
@st.cache_resource
def cross(): return json.load(open(MODELS / 'cross_calibration.json'))['pairs']
@st.cache_resource
def examples(): return json.load(open(MODELS / 'examples.json'))
@st.cache_resource
def diag(t):
    p = MODELS / 'diagnostics' / f'{t}.npz'
    return dict(np.load(p, allow_pickle=True)) if p.exists() else None
@st.cache_resource
def literature(): return pd.read_parquet(MODELS / 'literature.parquet')
IDX = index()
MAIN = ['Li_capacity_mg_g', 'Li_qmax_Langmuir_mg_g', 'Langmuir_KL_L_mg', 'k1_pseudo_first_order_1_min', 'k2_pseudo_second_order_g_mg_min', 'qe_kinetic_mg_g', 'Kd_distribution_coeff',
        'alpha_Li_Mg', 'alpha_Li_Na', 'alpha_Li_K', 'alpha_Li_Ca', 'Mn_dissolution_percent']
EXTRA = ['Freundlich_KF', 'Freundlich_n', 'Li_recovery_percent', 'Li_removal_percent', 'Li_desorption_recovery_percent', 'capacity_retention_percent', 'Ti_dissolution_percent']
NAME = {'Li_capacity_mg_g': 'Li capacity', 'Li_qmax_Langmuir_mg_g': 'Langmuir qₘₐₓ', 'Langmuir_KL_L_mg': 'Langmuir Kₗ', 'k1_pseudo_first_order_1_min': 'k₁ (pseudo-first order)',
        'k2_pseudo_second_order_g_mg_min': 'k₂ (pseudo-second order)', 'qe_kinetic_mg_g': 'Kinetic qₑ', 'Kd_distribution_coeff': 'Kd (distribution coefficient)', 'alpha_Li_Mg': 'α Li/Mg', 'alpha_Li_Na': 'α Li/Na',
        'alpha_Li_K': 'α Li/K', 'alpha_Li_Ca': 'α Li/Ca', 'Mn_dissolution_percent': 'Mn dissolution', 'Freundlich_KF': 'Freundlich KF', 'Freundlich_n': 'Freundlich n', 'Li_recovery_percent': 'Li recovery',
        'Li_removal_percent': 'Li removal', 'Li_desorption_recovery_percent': 'Desorption recovery', 'capacity_retention_percent': 'Capacity retention', 'Ti_dissolution_percent': 'Ti dissolution'}
UNIT = {'Li_capacity_mg_g': 'mg/g', 'Li_qmax_Langmuir_mg_g': 'mg/g', 'Langmuir_KL_L_mg': 'L/mg', 'k1_pseudo_first_order_1_min': '1/min', 'k2_pseudo_second_order_g_mg_min': 'g/(mg·min)',
        'qe_kinetic_mg_g': 'mg/g', 'Kd_distribution_coeff': 'mL/g', 'alpha_Li_Mg': '–', 'alpha_Li_Na': '–', 'alpha_Li_K': '–', 'alpha_Li_Ca': '–', 'Mn_dissolution_percent': '%', 'Freundlich_KF': '(mg/g)(L/mg)^1/n',
        'Freundlich_n': '–', 'Li_recovery_percent': '%', 'Li_removal_percent': '%', 'Li_desorption_recovery_percent': '%', 'capacity_retention_percent': '%', 'Ti_dissolution_percent': '%'}
PRETTY = {'contact_time_min': 'contact time', 'log_time': 'log contact time', 'Li_mg_L': 'initial Li (C₀)', 'log_Li': 'log C₀', 'Li_mmol_L': 'C₀ (mmol/L)', 'Li_per_dose_mg_g': 'Li per gram of sorbent (C₀/dose)',
          'log_Li_per_dose': 'log(C₀/dose)', 'adsorbent_dose_g_L': 'sorbent dose', 'log_dose': 'log dose', 'initial_pH': 'initial pH', 'temperature_C': 'temperature', 'Mg_mg_L': 'Mg', 'Na_mg_L': 'Na', 'K_mg_L': 'K',
          'Ca_mg_L': 'Ca', 'Mg_Li_ratio': 'Mg/Li ratio', 'log_Mg_Li': 'log Mg/Li', 'Na_Li_ratio': 'Na/Li ratio', 'K_Li_ratio': 'K/Li ratio', 'Ca_Li_ratio': 'Ca/Li ratio', 'sum_cations_mg_L': 'sum of cations',
          'cation_competition_index': 'competition index', 'BET_m2_g': 'BET area', 'cycle_number': 'cycle number', 'material_class': 'material class', 'material_subtype': 'material subtype',
          'material_family': 'sorbent family', 'mechanism': 'uptake mechanism', 'desorption_time_min': 'desorption time', 'desorption_conc_M': 'desorption concentration', 'calcination_temp_C': 'calcination temperature',
          'acid_treatment_conc_M': 'acid concentration', 'dopant_content_num': 'dopant content', 'active_content_percent': 'active content', 'particle_size_um': 'particle size', 'pore_size_nm': 'pore size'}
def nice(c): return PRETTY.get(c, c.replace('_', ' '))
def tier(m):
    r = m['heldout']
    return ('good', 'usable for new studies') if r >= 0.4 else (('fair', 'rough guide only') if r >= 0.15 else ('poor', 'not reliable for new studies'))
def disp(m, v): return 10 ** v - m.get('log_offset', 0) if m['log'] else v
def model_scale(m, v): return np.log10(v + m.get('log_offset', 0)) if m['log'] else v
def fmt(x):
    if x is None or not np.isfinite(x): return '–'
    a = abs(x); return f'{x:.3g}' if (a >= 0.01 or a == 0) and a < 1e4 else f'{x:.2e}'

# ------------------------------------------------------------------ input form
NUM, CAT, TXT = 'num', 'cat', 'text'
GROUPS = {
 'Sorbent': [('X__material_family', 'Sorbent family', CAT, 'Mn_LMO = spinel Mn oxide (HMO); Ti_HTO_LTO = H2TiO3 / Li4Ti5O12 derived; Al_LDH = Li/Al layered double hydroxide'),
             ('X__material_name_reported', 'Sample name (e.g. HMO, H2TiO3, Al-doped HMO)', TXT, 'used for keyword features (dopant, composite, series)'),
             ('X__adsorbent_form', 'Form (powder, granule, bead, membrane, foam, fibre)', TXT, None), ('X__binder_or_support', 'Binder or support (none, PVC, PVDF, chitosan, PAN)', TXT, None),
             ('X__active_content_percent', 'Active sieve content (wt%)', NUM, 'for composites'), ('X__dopant', 'Dopant (none, Al, Fe, Mg, Co, Zr, F, S)', TXT, None), ('X__dopant_content', 'Dopant content (e.g. 5 mol%)', TXT, None)],
 'Synthesis and texture': [('X__synthesis_route', 'Synthesis route (free text)', TXT, 'e.g. hydrothermal 180 °C 12 h; solid-state; sol-gel'), ('X__calcination_temp_C', 'Calcination temperature (°C)', NUM, None),
             ('X__acid_treatment_agent', 'Delithiation acid (HCl, H2SO4, HNO3)', TXT, None), ('X__acid_treatment_conc_M', 'Acid concentration (mol/L)', NUM, None), ('X__BET_m2_g', 'BET area (m²/g)', NUM, None),
             ('X__pore_size_nm', 'Pore size (nm)', NUM, None), ('X__particle_size_um', 'Particle size (µm)', NUM, None)],
 'Feed solution': [('X__Li_mg_L', 'Initial Li, C₀ (mg/L)', NUM, None), ('X__Na_mg_L', 'Na (mg/L)', NUM, None), ('X__K_mg_L', 'K (mg/L)', NUM, None), ('X__Mg_mg_L', 'Mg (mg/L)', NUM, None),
             ('X__Ca_mg_L', 'Ca (mg/L)', NUM, None), ('X__Cl_mg_L', 'Cl (mg/L)', NUM, None), ('X__SO4_mg_L', 'SO₄ (mg/L)', NUM, None), ('X__TDS_mg_L', 'TDS (mg/L)', NUM, None)],
 'Operating conditions': [('X__initial_pH', 'Initial pH', NUM, None), ('X__temperature_C', 'Temperature (°C)', NUM, None), ('X__contact_time_min', 'Contact time (min)', NUM, None),
             ('X__adsorbent_dose_g_L', 'Sorbent dose (g/L)', NUM, None), ('X__solution_volume_mL', 'Solution volume (mL)', NUM, None), ('X__cycle_number', 'Cycle number (1 = fresh)', NUM, None),
             ('X__desorption_agent', 'Desorption agent (HCl, H2SO4, water)', TXT, None), ('X__desorption_conc_M', 'Desorption concentration (mol/L)', NUM, None)]}
FIELDS = [f for g in GROUPS.values() for f, *_ in g]
RAW_VOCAB = bundle('Li_capacity_mg_g')['raw_vocab']
if 'vals' not in st.session_state: st.session_state.vals = dict(examples()[0]['values'])
if 'meas' not in st.session_state: st.session_state.meas = {}

st.title('LiSorb-Cal · calibrated predictions for lithium-selective sorbents')
st.caption(f"Trained on {IDX['rows']:,} experiments from {IDX['papers']} publications. Every reliability number shown here was measured on publications the model never saw. "
           "Most of the error of a literature model is a shift shared by a whole study, so one measurement of your own makes the predictions much more useful.")

with st.sidebar:
    st.header('1 · Describe your sorbent and test')
    ex = examples(); labels = [e['label'] for e in ex]
    pick = st.selectbox('Start from a literature example', labels, index=0, help='Fills the form with the conditions of a published experiment; edit anything afterwards.')
    if st.button('Load example', width='stretch'):
        st.session_state.vals = dict(ex[labels.index(pick)]['values']); st.session_state.meas = {}
        for f in FIELDS: st.session_state.pop('in_' + f, None)
        st.rerun()
    vals = {}
    for gname, items in GROUPS.items():
        with st.expander(gname, expanded=gname in ('Sorbent', 'Operating conditions', 'Feed solution')):
            for f, lab, kind, hlp in items:
                cur = st.session_state.vals.get(f)
                if kind == CAT:
                    opts = ['(unknown)'] + RAW_VOCAB.get(f, [])
                    v = st.selectbox(lab, opts, index=opts.index(cur) if cur in opts else 0, key='in_' + f, help=hlp); vals[f] = None if v == '(unknown)' else v
                elif kind == TXT:
                    v = st.text_input(lab, value='' if cur in (None, 'nan') else str(cur), key='in_' + f, help=hlp); vals[f] = v.strip() or None
                else:
                    v = st.text_input(lab, value='' if cur is None else f'{cur:g}' if isinstance(cur, (int, float)) else str(cur), key='in_' + f, help=hlp or 'leave empty if unknown')
                    try: vals[f] = float(v) if v.strip() else None
                    except ValueError: vals[f] = None; st.warning(f'{lab}: not a number, treated as unknown')
    for f, v in (st.session_state.vals or {}).items():
        if f not in vals: vals[f] = v
    st.session_state.vals_now = vals

# ------------------------------------------------------------------ core functions
def X_for(t, recs): B = bundle(t); return BF.features(recs, B['features'], B['cats'], B['vocab'])
def offset(t):
    """Eq. (1) of the paper from the stored measurements of this target, or cross-property from a measured capacity."""
    B = bundle(t); m = B['meta']; mu, lam = m.get('mu_hat', 0.0), m['lambda']; res = st.session_state.meas.get(t, [])
    if res: return mu + sum(r - mu for r in res) / (len(res) + lam), len(res), None
    for c in cross().get(t, []):
        rs = st.session_state.meas.get(c['source'], [])
        if rs:
            mS = bundle(c['source'])['meta']; ks = len(rs); g = c['gamma'] * (1 + mS['lambda']) / (1 + mS['lambda'] / ks)
            return mu + g * (float(np.mean(rs)) - mS.get('mu_hat', 0.0)), 0, c['source']
    return 0.0, 0, None
@st.cache_data(max_entries=5000, show_spinner=False)
def raw_pred(t, key):
    recs = json.loads(key); B = bundle(t); return B['model'].predict(X_for(t, recs))
def predict(t, recs=None, calibrated=True):
    recs = recs or [st.session_state.vals_now]; X = None; p = raw_pred(t, json.dumps(recs, sort_keys=True, default=str))
    off, k, src = offset(t) if calibrated else (0.0, 0, None)
    return p + off, k, src, X
def interval(t, pc, k):
    m = bundle(t)['meta']; hw = m['pi_halfwidth'].get(str(min(k, 3))); cov = m['pi_coverage'].get(str(min(k, 3)))
    if hw is None: return None, None, None
    lo, hi = pc - hw, pc + hw
    if not m['log']: lo = np.maximum(lo, 0)
    return disp(m, lo), disp(m, hi), cov
def domain(t, X):
    a = bundle(t)['ad']; Z = a['prep'].transform(X); d = a['nn'].kneighbors(Z, 5)[0].mean(1); Zt = np.c_[np.ones(len(Z)), Z]
    h = np.einsum('ij,jk,ik->i', Zt, a['G'], Zt); idx = a['nn'].kneighbors(Z, 10)[1][0]
    return d, a['threshold'], h, a['hstar'], a['rows'][idx]

tabs = st.tabs(['📊 Predict', '🎯 Calibrate with your data', '📈 What-if curves', '🔍 Why this prediction', '🧭 Applicability domain', '📚 Literature neighbours', '📄 Batch', '📐 How good is the model?', 'ℹ️ About the models'])

# ------------------------------------------------------------------ Predict
with tabs[0]:
    fam = st.session_state.vals_now.get('X__material_family')
    if fam and fam not in bundle('Li_capacity_mg_g')['meta']['families_trained']: st.error('This sorbent family is not in the training data; the models do not transfer between chemistries.')
    if not fam: st.info('Choose a sorbent family in the sidebar. The models do not transfer between families (paper, Section 3.3).')
    show = st.radio('Properties', ['12 main properties', 'all 19 properties'], horizontal=True)
    TG = MAIN + (EXTRA if show == 'all 19 properties' else [])
    cols = st.columns(3); rows = []
    for i, t in enumerate(TG):
        B = bundle(t); m = B['meta']; pc, k, src, X = predict(t); lo, hi, cov = interval(t, pc[0], k); cls, txt = tier(m)
        v = disp(m, pc[0]); rows.append(dict(property=NAME[t], unit=UNIT[t], prediction=v, lower90=lo, upper90=hi, calibrated='yes' if (k or src) else 'no', reliability=txt))
        with cols[i % 3]:
            st.markdown(f"<div class='card'><b>{NAME[t]}</b> <span class='badge {cls}'>{txt}</span><br>"
                        f"<span style='font-size:1.5rem'>{fmt(v)}</span> {UNIT[t]}<br>"
                        f"<span style='color:#666;font-size:0.85rem'>90% interval {fmt(lo)} – {fmt(hi)}" + (f" · coverage on new studies {cov:.2f}" if cov else '') + "<br>"
                        f"R² on unseen publications {m['heldout']:.2f}" + (f" · calibrated with {k} measurement(s)" if k else (f" · calibrated from your {NAME[src]}" if src else '')) + "</span></div>", unsafe_allow_html=True)
    st.download_button('Download these predictions (CSV)', pd.DataFrame(rows).to_csv(index=False).encode(), 'lisorb_predictions.csv')
    st.caption('Badges: R² ≥ 0.40 on unseen publications → usable for new studies; 0.15–0.40 → rough guide; below 0.15 → not reliable without a measurement of the same property. Intervals are empirical (no coverage guarantee).')

# ------------------------------------------------------------------ Calibrate
with tabs[1]:
    st.markdown("Enter a value you measured for **your** material. The app estimates how far your study sits from the literature average and shifts every prediction of that property by a shrunk amount, "
                "ĉ = μ̂ + Σ(r − μ̂)/(k + λ) (Eq. 1 of the paper). A measured capacity also calibrates kinetic qₑ and Langmuir qₘₐₓ, the pairs that worked in the paper.")
    c1, c2, c3 = st.columns([1.3, 1, 1])
    t_m = c1.selectbox('Property you measured', MAIN + EXTRA, format_func=lambda t: f'{NAME[t]} ({UNIT[t]})')
    val = c2.number_input('Measured value', min_value=0.0, value=0.0, format='%.4g')
    c3.write(''); c3.write('')
    if c3.button('Add measurement', width='stretch'):
        if val > 0:
            m = bundle(t_m)['meta']; p0 = raw_pred(t_m, json.dumps([st.session_state.vals_now], sort_keys=True, default=str))[0]
            st.session_state.meas.setdefault(t_m, []).append(float(model_scale(m, val) - p0)); st.toast('Measurement stored (conditions as in the sidebar).'); st.rerun()
        else: st.warning('Enter a positive value.')
    if st.button('Clear all measurements'): st.session_state.meas = {}
    if st.session_state.meas:
        tab = []
        for t in MAIN:
            m = bundle(t)['meta']; p_unc = predict(t, calibrated=False)[0][0]; pc, k, src, _ = predict(t); lo0, hi0, _ = interval(t, p_unc, 0); lo1, hi1, _ = interval(t, pc[0], k)
            if k or src: tab.append({'Property': NAME[t], 'Before': fmt(disp(m, p_unc)), '90% interval before': f'{fmt(lo0)} – {fmt(hi0)}', 'After': fmt(disp(m, pc[0])),
                                     '90% interval after': f'{fmt(lo1)} – {fmt(hi1)}', 'Source': f'{k} own point(s)' if k else f'cross-property from {NAME[src]}'})
        st.dataframe(pd.DataFrame(tab), hide_index=True, width='stretch')
        st.info('What the paper found: after one measurement, the calibrated model beat the measured value alone for Li capacity and Mn dissolution. For the fitted constants (Kₗ, k₁, k₂) '
                'the measured value alone was as good, so treat calibrated constants as the measured value plus a small correction.')
    else: st.caption('No measurement stored yet.')

# ------------------------------------------------------------------ What-if
with tabs[2]:
    c1, c2 = st.columns(2)
    t_w = c1.selectbox('Property', MAIN, format_func=lambda t: NAME[t], key='wt')
    rng_ = bundle(t_w).get('ranges', {}); SW = {'X__Li_mg_L': 'Initial Li, C₀ (mg/L)', 'X__adsorbent_dose_g_L': 'Sorbent dose (g/L)', 'X__contact_time_min': 'Contact time (min)', 'X__initial_pH': 'Initial pH', 'X__temperature_C': 'Temperature (°C)'}
    var = c2.selectbox('Condition to vary', [k for k in SW if k in rng_], format_func=lambda k: SW[k])
    lo_, hi_ = rng_[var]; logx = var in ('X__Li_mg_L', 'X__adsorbent_dose_g_L', 'X__contact_time_min')
    grid = np.geomspace(max(lo_, 1e-3), hi_, 40) if logx else np.linspace(lo_, hi_, 40)
    recs = [dict(st.session_state.vals_now, **{var: float(g)}) for g in grid]; m = bundle(t_w)['meta']
    pc, k, src, _ = predict(t_w, recs); lo, hi, _ = interval(t_w, pc, k)
    fig = go.Figure()
    if lo is not None: fig.add_trace(go.Scatter(x=np.r_[grid, grid[::-1]], y=np.r_[hi, lo[::-1]], fill='toself', fillcolor='rgba(42,120,214,0.15)', line=dict(width=0), name='90% interval'))
    fig.add_trace(go.Scatter(x=grid, y=disp(m, pc), mode='lines', line=dict(color='#2a78d6', width=3), name='prediction' + (' (calibrated)' if (k or src) else '')))
    cur = st.session_state.vals_now.get(var)
    if cur: fig.add_vline(x=cur, line_dash='dot', line_color='#888')
    fig.update_layout(xaxis_title=SW[var], yaxis_title=f'{NAME[t_w]} ({UNIT[t_w]})', height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_type='log' if logx else 'linear', yaxis_type='log' if m['log'] else 'linear')
    st.plotly_chart(fig, width='stretch')
    st.caption('Range: 5th–95th percentile of the training data. All other inputs stay as in the sidebar. The curve shows what the model learned across the literature; '
               'within a single study such trends are weaker and can even reverse (paper, Section 3.2), so use curves to plan experiments, not to replace them.')

# ------------------------------------------------------------------ Explain
with tabs[3]:
    t_e = st.selectbox('Property', MAIN, format_func=lambda t: NAME[t], key='et')
    B = bundle(t_e); X = X_for(t_e, [st.session_state.vals_now])
    exact = st.toggle('Exact SHAP of the whole ensemble (slower)', value=False)
    with st.spinner('Computing contributions...'):
        if exact:
            import shap
            tot = np.zeros(X.shape[1])
            for w, ms in B['model'].members:
                for mm in ms:
                    est = mm.m if hasattr(mm, 'm') else mm; Xi = X.fillna(-999) if hasattr(mm, 'm') else X
                    tot += w * np.asarray(shap.TreeExplainer(est).shap_values(Xi))[0] / len(ms)
            contrib = pd.Series(tot, index=B['features']); src_ = 'stacked ensemble'
        else:
            contrib = pd.Series(B['lgb'].predict(X, pred_contrib=True)[0][:-1], index=B['features']); src_ = 'LightGBM component'
    top = contrib[contrib.abs() > 0].sort_values(key=np.abs, ascending=False).head(12)[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=[nice(i) for i in top.index], orientation='h', marker_color=['#d43c3c' if v < 0 else '#1a7f5a' for v in top.values]))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), xaxis_title='Contribution to the prediction (modelling scale)', title=f'Largest contributions ({src_})')
    st.plotly_chart(fig, width='stretch')
    st.caption('Contributions describe the model, not the sorbent: they show which inputs moved this prediction, not what would happen if you changed them in the laboratory (paper, Section 3.6).')

# ------------------------------------------------------------------ Applicability domain
with tabs[4]:
    t_a = st.selectbox('Property', MAIN, format_func=lambda t: NAME[t], key='at')
    X = X_for(t_a, [st.session_state.vals_now]); d, thr, h, hs, _ = domain(t_a, X)
    c1, c2, c3 = st.columns(3)
    c1.metric('Distance to the 5 nearest training rows', f'{d[0]:.2f}', delta=f'95th percentile of training: {thr:.2f}', delta_color='off')
    c2.metric('Leverage h', f'{h[0]:.3g}', delta=f'warning leverage h* = {hs:.3g}', delta_color='off')
    c3.metric('Verdict', 'typical inputs' if (d[0] <= thr and h[0] <= hs) else 'unusual inputs')
    meas_res = st.session_state.meas.get(t_a, []); m = bundle(t_a)['meta']
    fig = go.Figure(); fig.add_vline(x=hs, line_dash='dash', line_color='#d43c3c'); fig.add_hline(y=3, line_dash='dot', line_color='#888'); fig.add_hline(y=-3, line_dash='dot', line_color='#888')
    Dg = diag(t_a)
    if Dg is not None:
        fig.add_trace(go.Scattergl(x=Dg['lev'], y=Dg['sres'], mode='markers', marker=dict(size=3, color='#9db8dc', opacity=0.45), name='literature rows, unseen publications',
                                   hovertemplate='h %{x:.3g}<br>residual %{y:.2f}<extra></extra>'))
    yv = (meas_res[-1] / m['nested_rmse']) if meas_res else 0.0
    fig.add_trace(go.Scatter(x=[h[0]], y=[yv], mode='markers', marker=dict(size=14, color='#2a78d6', symbol='star'), name='your input'))
    fig.update_layout(xaxis_type='log', xaxis_title='Leverage h', yaxis_title='Standardised residual (needs a measured value)', height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_range=[-6, 6],
                      title='Williams plot: your input among the literature rows (nested evaluation)', legend=dict(orientation='h', y=-0.2))
    st.plotly_chart(fig, width='stretch')
    st.info('In the paper, distance and leverage flagged unusual inputs but predicted the error only weakly: for capacity not at all (Spearman ρ ≈ 0), for the separation factors a little (ρ ≈ 0.2). '
            'Most of the error belongs to the study, not to the inputs, so a familiar-looking input can still be far off. Use this as novelty information.')

# ------------------------------------------------------------------ Literature neighbours
with tabs[5]:
    t_l = st.selectbox('Property', MAIN, format_func=lambda t: NAME[t], key='lt')
    X = X_for(t_l, [st.session_state.vals_now]); _, _, _, _, rows_ = domain(t_l, X); L = literature()
    nb = L.loc[[r for r in rows_ if r in L.index]].copy(); col = 'y__' + t_l
    out = pd.DataFrame({'Publication': nb.meta__paper_short, 'Year': nb.meta__year, 'DOI': nb.meta__doi, 'Material': nb.X__material_name_reported, 'C₀ (mg/L)': nb.X__Li_mg_L,
                        'Dose (g/L)': nb.X__adsorbent_dose_g_L, 'Time (min)': nb.X__contact_time_min, 'pH': nb.X__initial_pH, f'Measured {NAME[t_l]} ({UNIT[t_l]})': pd.to_numeric(nb[col], errors='coerce')})
    st.markdown('The ten most similar experiments in the training data (standardised numeric inputs). Several often come from one study, which is the pattern behind the publication offset.')
    st.dataframe(out, hide_index=True, width='stretch')
    st.caption(f"{out['Publication'].nunique()} distinct publications among the ten neighbours.")

# ------------------------------------------------------------------ Batch
with tabs[6]:
    t_b = st.selectbox('Property', MAIN + EXTRA, format_func=lambda t: NAME[t], key='bt')
    st.markdown('Upload a CSV with one experiment per row (columns as in the template). Optional columns `material_id` and `measured` calibrate each material with its own measured rows.')
    st.download_button('Download template', pd.DataFrame(columns=FIELDS + ['material_id', 'measured']).to_csv(index=False).encode(), 'lisorb_template.csv')
    up = st.file_uploader('CSV file', type='csv')
    if up is not None:
        df = pd.read_csv(up); recs = [{f: (None if pd.isna(r.get(f, np.nan)) else r.get(f)) for f in FIELDS} for _, r in df.iterrows()]
        B = bundle(t_b); m = B['meta']; X = X_for(t_b, recs); p = B['model'].predict(X); out = df.copy(); out['prediction'] = disp(m, p)
        if {'measured', 'material_id'} <= set(df.columns):
            r_ = pd.Series([model_scale(m, v) if pd.notna(v) and v > 0 else np.nan for v in df.measured]) - p; g = r_.groupby(df.material_id).agg(['sum', 'count'])
            mu = m.get('mu_hat', 0.0); g['c'] = mu + (g['sum'] - g['count'] * mu) / (g['count'] + m['lambda']); out['prediction_calibrated'] = disp(m, p + df.material_id.map(g['c']).fillna(0).values)
        st.dataframe(out, width='stretch'); st.download_button('Download results', out.to_csv(index=False).encode(), f'lisorb_{t_b}.csv')

# ------------------------------------------------------------------ Model quality (parity)
with tabs[7]:
    st.markdown("Each point is one literature experiment. **Random split**: the model had seen other rows of the same publication. **Unseen publication**: the whole publication was held out, which is the situation of a new laboratory. "
                "The gap between the two is the main finding of the paper.")
    c1, c2 = st.columns([1.2, 1])
    t_q = c1.selectbox('Property', MAIN, format_func=lambda t: NAME[t], key='qt'); view = c2.radio('Show', ['both', 'unseen publications only', 'random split only'], horizontal=True)
    Dg = diag(t_q); m = bundle(t_q)['meta']
    if Dg is None: st.info('Diagnostics not available for this property.')
    else:
        lab_ = 'log₁₀ ' if m['log'] else ''
        fig = go.Figure(); allv = [Dg['y'], Dg['pred_nested']] + ([Dg['pred_random']] if 'pred_random' in Dg else [])
        lo_, hi_ = float(np.nanpercentile(np.concatenate(allv), 0.5)), float(np.nanpercentile(np.concatenate(allv), 99.5))
        def r2(a, b): return 1 - np.nansum((a - b) ** 2) / np.nansum((a - np.nanmean(a)) ** 2)
        if 'pred_random' in Dg and view != 'unseen publications only':
            fig.add_trace(go.Scattergl(x=Dg['y'], y=Dg['pred_random'], mode='markers', marker=dict(size=4, color='#2a78d6', opacity=0.35), name=f"random split, R² {r2(Dg['y'], Dg['pred_random']):.2f}"))
        if view != 'random split only':
            fig.add_trace(go.Scattergl(x=Dg['y'], y=Dg['pred_nested'], mode='markers', marker=dict(size=4, color='#eb6834', opacity=0.35), name=f"unseen publication, R² {r2(Dg['y'], Dg['pred_nested']):.2f}"))
        fig.add_trace(go.Scatter(x=[lo_, hi_], y=[lo_, hi_], mode='lines', line=dict(color='#555', dash='dash'), name='1:1'))
        res = st.session_state.meas.get(t_q, [])
        if res:
            p0_ = raw_pred(t_q, json.dumps([st.session_state.vals_now], sort_keys=True, default=str))[0]; pc_, k_, _, _ = predict(t_q)
            fig.add_trace(go.Scatter(x=[p0_ + res[-1]], y=[p0_], mode='markers', marker=dict(size=14, color='#d43c3c', symbol='star'), name='your measurement vs. prediction'))
            fig.add_trace(go.Scatter(x=[p0_ + res[-1]], y=[pc_[0]], mode='markers', marker=dict(size=14, color='#1a7f5a', symbol='star'), name='after calibration'))
        fig.update_layout(height=520, xaxis_title=f'Measured {lab_}{NAME[t_q]}', yaxis_title=f'Predicted {lab_}{NAME[t_q]}', xaxis_range=[lo_, hi_], yaxis_range=[lo_, hi_], margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(fig, width='stretch')
        st.caption('Points of one publication often sit together above or below the line: that shared shift is the publication offset, and it is what one measurement of your own removes.')

# ------------------------------------------------------------------ About
with tabs[8]:
    st.markdown("**What the models can and cannot do** (from the nested, publication-grouped evaluation of the paper)")
    card = []
    for t in MAIN + EXTRA:
        m = bundle(t)['meta']; card.append({'Property': NAME[t], 'Rows': m['rows'], 'Publications': m['papers'], 'R², random split': round(m['cv'], 2), 'R², unseen publications': round(m['heldout'], 2),
                                            'Offset share': round(m.get('offset_share', np.nan), 2), 'λ': round(m['lambda'], 2), 'Reliability': tier(m)[1]})
    st.dataframe(pd.DataFrame(card), hide_index=True, width='stretch')
    st.markdown("""
* A random split of literature rows scores interpolation; the R² on unseen publications is the honest figure for a new study.
* About half to nine-tenths of the error is a **publication offset** shared by all rows of a study. One measurement of your own removes much of it.
* The models should be used **within the three sorbent families** they were trained on (Mn oxides, Ti oxides, Li/Al layered double hydroxides).
* Rankings of similar materials within one study are close to chance; calibration cannot change them.
* Predictions are statistical estimates from published data and do not replace measurement.""")
    st.caption('Model files, SHA-256 hashes and stacking weights: models/deploy_manifest.json. Code and data: see the repository of the paper.')
