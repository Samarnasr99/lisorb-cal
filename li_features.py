"""
Lithium DLE - prediction of Li adsorption capacity (y__Li_capacity_mg_g, mg/g)
Reproducible pipeline: data completion -> row curation -> feature engineering -> label-consistency filter
-> tuned LightGBM + XGBoost + ExtraTrees blend -> evaluation (train, 3x5-fold CV, 80/20 holdout, paper-grouped CV).
usage:  python li_capacity_pipeline.py Lithium_DLE_V11_1.xlsx --out li_capacity_model_out
"""
import re, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')


TARGET = 'y__Li_capacity_mg_g'
NUM_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def parse_num(v):
    """numeric or text -> float. ranges 'a-b' -> midpoint; '<x' -> x; leading number otherwise."""
    if v is None or (isinstance(v, float) and np.isnan(v)): return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)): return float(v)
    s = str(v).strip().replace('–', '-').replace(',', '')
    m = re.match(r'^\s*(\d*\.?\d+)\s*-\s*(\d*\.?\d+)', s)
    if m: return (float(m.group(1)) + float(m.group(2))) / 2
    m = NUM_RE.search(s)
    return float(m.group()) if m else np.nan

NUMERIC = ['X__active_content_percent','X__particle_size_um','X__BET_m2_g','X__pore_size_nm','X__pore_volume_cm3_g',
 'X__calcination_temp_C','X__acid_treatment_conc_M','X__Li_mg_L','X__Na_mg_L','X__K_mg_L','X__Mg_mg_L','X__Ca_mg_L',
 'X__Cl_mg_L','X__SO4_mg_L','X__B_mg_L','X__Sr_mg_L','X__TDS_mg_L','X__initial_pH','X__temperature_C',
 'X__contact_time_min','X__adsorbent_dose_g_L','X__solution_volume_mL','X__agitation_rpm','X__flow_rate_mL_min',
 'X__bed_volume_mL','X__cycle_number','X__desorption_conc_M','X__desorption_temp_C','X__desorption_time_min',
 'D__Mg_Li_ratio','D__Na_Li_ratio','D__K_Li_ratio','D__Ca_Li_ratio','D__divalent_Li_ratio',
 'D__cation_competition_index','D__sum_cations_mg_L','D__log_Mg_Li']

def low(s): return s.fillna('').astype(str).str.lower()

def kw(series, rules, default='other', na='missing'):
    s = low(series); out = pd.Series(default, index=series.index)
    out[series.isna()] = na
    done = pd.Series(False, index=series.index) | series.isna()
    for label, pats in rules:
        hit = s.str.contains(pats, regex=True) & ~done
        out[hit] = label; done |= hit
    return out

def build(df):
    F = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        F[c.split('__', 1)[1]] = df[c].map(parse_num)
    # cycle numbers like '0-600' fine. dopant content numeric where parsable
    F['dopant_content_num'] = df['X__dopant_content'].map(parse_num)

    # ---------- physically-motivated engineered features ----------
    F['Li_per_dose_mg_g'] = F['Li_mg_L'] / F['adsorbent_dose_g_L']          # C0/(m/V): theoretical upper bound of q
    F['log_Li_per_dose'] = np.log10(F['Li_per_dose_mg_g'].clip(lower=1e-3))
    F['log_Li'] = np.log10(F['Li_mg_L'].clip(lower=1e-3))
    F['log_time'] = np.log10(F['contact_time_min'].clip(lower=0.1))
    F['log_dose'] = np.log10(F['adsorbent_dose_g_L'].clip(lower=1e-3))
    F['Li_mmol_L'] = F['Li_mg_L'] / 6.941
    F['BET_x_porevol'] = F['BET_m2_g'] * F['pore_volume_cm3_g']

    name = low(df['X__material_class']) + ' | ' + low(df['X__material_name_reported'])
    F['material_family'] = df['X__material_family'].fillna('missing').replace({'Ti_LTO':'Ti_HTO_LTO','other_oxide':'Other'})
    F['material_subtype'] = kw(name, [
        ('LDH', r'ldh|hydrotalcite|lial|li/al|aluminum hydroxide|al\(oh\)'),
        ('LTO_spinel', r'ti5o12|lto|li4ti5'),
        ('HTO_layered', r'h2tio3|li2tio3|hto|titan'),
        ('LMO_1.6', r'1\.6mn1\.6|mn1\.6o4|li1\.6|h1\.6|mno2·0\.5|mno2\.0\.5|0\.5h2o'),
        ('LMO_1.33', r'1\.33|mn1\.67|li4mn5o12|h4mn5o12'),
        ('LMO_2', r'limn2o4|lambda|λ|mn2o4|hmn2o4'),
        ('LMO_other', r'mn|manganese|hmo|lmo'),
        ('FePO4_olivine', r'fepo4|lifepo4|olivine'),
        ('aluminosilicate', r'alsi|zeolite|silicate'),
        ('MOF_polymer', r'mof|crown|polymer|resin|imprint'),
    ])
    F['material_class'] = df['X__material_class'].fillna('missing')
    F['material_name'] = df['X__material_name_reported'].fillna('missing')
    F['form'] = kw(df['X__adsorbent_form'], [
        ('powder', r'powder|nanopart|nanorod|nanosheet|crystal|particle'),
        ('granule', r'granul|bead|sphere|pellet|ball'),
        ('membrane_fiber', r'membrane|fiber|fibre|film|mat|cloth'),
        ('electrode', r'electrode'),
        ('foam_monolith', r'foam|monolith|sponge|aerogel|hydrogel|scaffold|column'),
    ])
    F['is_composite'] = (df['X__binder_or_support'].notna() & ~low(df['X__binder_or_support']).str.match(r'^(none|n/a|no )')).astype(int)
    F['binder'] = kw(df['X__binder_or_support'], [
        ('none', r'^none|^no |n/a'),
        ('PVC', r'pvc'), ('PVDF', r'pvdf'), ('PVA', r'pva|polyvinyl alcohol'), ('EVAL', r'eva'),
        ('chitosan_alginate_cellulose', r'chitosan|alginate|cellulose|agar|biopolymer'),
        ('carbon', r'carbon|graphene|biochar|cnt|char'),
        ('PSF_PES_PAN', r'psf|polysulfone|pes|pan|polyacrylonitrile'),
        ('inorganic_support', r'glass|silica|zro2|al2o3|ceramic|sio2|coating'),
    ])
    F['dopant'] = kw(df['X__dopant'], [
        ('none', r'^none|^no |undoped'),
        ('Zr', r'zr'), ('Al', r'^al|al,|al '), ('Fe', r'fe'), ('Ni', r'ni'), ('Co', r'co'), ('Mg', r'mg'),
        ('Nb', r'nb'), ('W', r'^w|w '), ('Cr', r'cr'), ('Cu', r'cu'), ('Zn', r'zn'), ('Ti', r'ti'),
        ('F_N_S', r'^f|^n|^s|fluor|nitrog|sulf'), ('K_Na', r'^k|na'),
    ], default='other_dopant', na='none')
    syn = low(df['X__synthesis_route'])
    for k, p in {'hydrothermal': r'hydrothermal|solvothermal', 'solid_state': r'solid[- ]state|ball[- ]mill|calcin',
                 'sol_gel': r'sol[- ]gel|citric|pechini', 'coprecip': r'co-?precip|precipitat',
                 'electrospin': r'electrospin', 'freeze_dry': r'freeze', 'microwave': r'microwave',
                 'template': r'template', 'coating': r'coat'}.items():
        F['syn_' + k] = syn.str.contains(p, regex=True).astype(int)
    F['acid'] = kw(df['X__acid_treatment_agent'], [
        ('HCl', r'hcl'), ('H2SO4', r'h2so4|sulfuric'), ('HNO3', r'hno3'),
        ('persulfate', r's2o8|persulf'), ('organic_acid', r'citric|malic|acetic|oxalic'), ('none', r'none|^no'),
    ], na='none_reported')
    F['mechanism'] = kw(df['X__mechanism_class'], [
        ('intercalation_LDH', r'interlayer|intercalation(?!.*redox)|ldh'),
        ('redox_electro', r'redox|electro|faradaic|capacitive'),
        ('ion_sieve', r'sieve|ion[_ ]?exchange|h\+|li\+/h\+|lih'),
        ('physisorption', r'physi|electrostatic|chelat|complex'),
    ])
    F['exp_type'] = kw(df['X__experiment_type'], [
        ('cycling', r'cycl|regenerat|reuse'), ('column', r'column|fixed|bed|flow'),
        ('electrochemical', r'electro'), ('pilot', r'pilot|field|industrial'), ('membrane', r'membrane'),
        ('batch', r'batch|kinetic|isotherm|equilib|static'),
    ])
    F['feed'] = kw(df['X__feed_matrix_complexity'], [
        ('real_brine', r'real|industrial|field|actual'), ('single_salt', r'single|simple|pure'),
        ('multi_salt', r'multi|simulat|synthetic|complex|mixed|competit'),
    ])
    F['brine'] = kw(df['X__brine_type'], [
        ('salt_lake', r'salt[_ ]lake|salar|qarhan|qinghai'), ('seawater', r'sea'),
        ('produced_water', r'produced|oilfield|shale|formation|gas|oil'), ('geothermal', r'geotherm'),
        ('simulated_brine', r'simulat'), ('battery_leachate', r'leach|battery|waste'),
        ('synthetic', r'synthetic|licl|lioh|li2so4|li2co3|solution|di'),
    ])
    F['desorb_agent'] = kw(df['X__desorption_agent'], [
        ('HCl', r'hcl'), ('H2SO4', r'h2so4|sulfur'), ('water', r'water|di |h2o'), ('LiCl', r'licl'),
        ('persulfate', r's2o8'), ('electro', r'electro|charge'),
    ], na='none_reported')
    return F

CATS = ['material_family','material_subtype','material_class','material_name','form','binder','dopant','acid',
        'mechanism','exp_type','feed','brine','desorb_agent']


# ---------------- text-mined synthesis descriptors (added in iteration 2) ----------------
def _first(s, pat, lo=None, hi=None, conv=None):
    out = s.str.extract(pat, flags=re.I)[0].map(lambda v: float(v) if v is not None and v == v else np.nan)
    if conv: out = conv(out)
    if lo is not None: out = out.where((out >= lo) & (out <= hi))
    return out

def synth_feats(df):
    syn = df['X__synthesis_route'].fillna('').astype(str)
    note = df['meta__extractor_note'].fillna('').astype(str)
    t = (syn + ' ; ' + note).str.replace('°', '', regex=False)
    G = pd.DataFrame(index=df.index)
    G['Li_M_molar_ratio'] = _first(t, r'li\s*/\s*(?:mn|ti|al|m)\s*(?:molar|mol)?\s*(?:ratio)?\s*(?:n\s*)?[=:~ ]?\s*(\d*\.?\d+)', 0.05, 10)
    G['hydrothermal_T_C'] = _first(t, r'(?:hydrothermal|autoclave|solvothermal)[^;|]{0,40}?(\d{2,3})\s*c\b', 60, 300)
    hk = _first(t, r'(?:hydrothermal|autoclave|solvothermal)[^;|]{0,40}?(\d{3})\s*k\b', 330, 600)
    G['hydrothermal_T_C'] = G['hydrothermal_T_C'].fillna(hk - 273)
    G['hydrothermal_time_h'] = _first(t, r'(?:hydrothermal|autoclave|solvothermal)[^;|]{0,50}?(\d+\.?\d*)\s*h\b', 0.5, 200)
    G['calcination_time_h'] = _first(t, r'calcin\w*[^;|]{0,35}?\d{3,4}\s*c\W{0,3}\s*(?:for\s*)?(\d+\.?\d*)\s*h\b', 0.5, 100)
    n = note.str.lower()
    G['basis'] = np.select([n.str.contains(r'per g(?:ram)? of (?:active|ion[- ]sieve|lis|hmo|hto|lmo|lto)|per-g-(?:hmo|hto|active)|active (?:mass|component) basis|normalized per (?:g|gram) of active'),
                            n.str.contains(r'per g(?:ram)? (?:of )?(?:composite|granule|bead|membrane|electrode)|composite basis|g/l of adsorbent bed')],
                           ['active', 'composite'], 'unspecified')
    G['unit_converted_mmol'] = n.str.contains('mmol/g').astype(int)
    return G

def precursor_feats(df):
    note = df['meta__extractor_note'].fillna('').str.lower()
    salts = note.str.findall(r'(?:mn|li|ti) (?:salt|source|precursor)\s*[=:]\s*[^,;|]+').str.join(' ')
    t = (df['X__synthesis_route'].fillna('').str.lower() + ' ; ' + salts + ' ; ' + df['X__material_name_reported'].fillna('').str.lower())
    tn = t + ' ; ' + note
    G = pd.DataFrame(index=df.index)
    G['Mn_source'] = np.select([t.str.contains(r'mncl2'), t.str.contains(r'mnso4'), t.str.contains(r'mn\(no3\)2|manganese nitrate'),
        t.str.contains(r'mn\(ch3coo\)2|mn acetate|manganese acetate|mn\(ac\)2'), t.str.contains(r'kmno4'), t.str.contains(r'mn2o3'),
        t.str.contains(r'mn3o4'), t.str.contains(r'mnco3'), t.str.contains(r'mnooh|γ-mnooh|gamma-mnooh'), t.str.contains(r'mno2|emd')],
        ['MnCl2','MnSO4','MnNO3','MnAc','KMnO4','Mn2O3','Mn3O4','MnCO3','MnOOH','MnO2'], 'na')
    G['Li_source'] = np.select([t.str.contains(r'li2b4o7'), t.str.contains(r'lioh'), t.str.contains(r'li2co3'), t.str.contains(r'lino3'),
        t.str.contains(r'ch3cooli|li acetate|lithium acetate|lic2h3o2'), t.str.contains(r'licl')],
        ['Li2B4O7','LiOH','Li2CO3','LiNO3','LiAc','LiCl'], 'na')
    G['Ti_source'] = np.select([t.str.contains(r'anatase'), t.str.contains(r'rutile'), t.str.contains(r'p25'),
        t.str.contains(r'tbot|butoxide|tetrabutyl'), t.str.contains(r'ttip|isopropoxide'), t.str.contains(r'ticl4|tiocl|tioso4|ti\(so4\)2'),
        t.str.contains(r'tio2')], ['anatase','rutile','P25','TBOT','TTIP','Ti_salt','TiO2'], 'na')
    G['oxidant'] = np.select([t.str.contains(r'\(nh4\)2s2o8|aps\b|ammonium persulf'), t.str.contains(r'k2s2o8|na2s2o8'),
        t.str.contains(r'h2o2'), t.str.contains(r'kmno4')], ['APS','K/Na_S2O8','H2O2','KMnO4'], 'na')
    G['cycling_row'] = tn.str.contains(r'cycl|regenerat|reus').astype(int)
    G['substrate_basis'] = tn.str.contains(r'per gram of al metal|per g of al|per gram of substrate|per m2|per cm2|g/l of adsorbent bed').astype(int)
    return G

def series_feats(df):
    """sample-series descriptors: numeric suffix in sample name (e.g. Ni-HTO-0.08, HMO-Ti-6, HTO-650) and
    calcination T/time stated in per-row notes for synthesis-series rows"""
    name = df['X__material_name_reported'].fillna('').astype(str)
    G = pd.DataFrame(index=df.index)
    G['name_series_value'] = name.str.extract(r'[-_ ](\d*\.?\d+)\s*(?:\)|%|wt%|h|c)?\s*(?:\(.*\))?\s*$', flags=re.I)[0].astype(float)
    note = df['meta__extractor_note'].fillna('').astype(str).str.lower().str.replace('°', '', regex=False)
    G['note_calc_T_C'] = _first(note, r'(?:calcin\w*|sinter\w*|series|annealed?)[^;|]{0,40}?(\d{3,4})\s*c\b', 150, 1300)
    G['note_calc_time_h'] = _first(note, r'(?:calcin\w*|sinter\w*|series|annealed?)[^;|]{0,40}?\d{3,4}\s*c\W{0,3}\s*(\d+\.?\d*)\s*h\b', 0.5, 100)
    G['note_ads_pH'] = _first(note, r'(?:at|optimum|optimal)\s*ph\s*(\d{1,2}(?:\.\d+)?)', 0, 14)
    return G

# ====== data completion ======
"""Data completion: (1) mine free-text notes for missing experimental conditions, (2) fill missing protocol
values within a paper when that paper reports a single unique value for the column."""

def txt(r):
    return ' ; '.join(str(r[c]) for c in ['meta__extractor_note','X__other_ions_note','X__brine_source'] if pd.notna(r[c]))

def f(x):
    try: return float(x)
    except: return np.nan

BAD_CTX = re.compile(r'desorb|desorp|elut|strip|pickl|wash|acid[- ]?treat|delith|regenerat|calcin|anneal|sinter|hydrotherm|solvotherm|dried|drying|oven|nmr|zeta|xps|leach|precursor|synthes|roast|co-?precip|ball|mill|aged|polymeri|initiator|dispersant|protonat|eluat|eluent|hcl|h2so4|hno3|not (?:reported|stated|numeric)|moved|max(?:imum)? |breakthrough|after \\d+ mg/l licl|enrich', re.I)

FWD_BAD = re.compile(r'^\s*(?:-\s*\d|–|to \d|\s*\d*\.?\d*\s*m\s*hcl|\s*\d*\.?\d*\s*m\s*h2so4|[^;|]{0,12}(?:hcl|acid|eluent|eluat|not reported))', re.I)

def _ok(t, m, back=45):
    """reject a match whose preceding context refers to synthesis, desorption or characterisation,
    or that is followed by a range ('pH 2-6') or an acid/eluent"""
    return not BAD_CTX.search(t[max(0, m.start() - back):m.start()]) and not FWD_BAD.search(t[m.end():m.end() + 25])

def mine(t, spans=False):
    out, sp = {}, {}
    def put(k, v, m): out[k] = v; sp[k] = t[max(0, m.start() - 40):m.end() + 10]
    # dose: 'x mg (adsorbent) in y mL'
    for m in re.finditer(r'(\d+\.?\d*)\s*(mg|g)\s*(?:of\s+)?(?:[\w\-\.\(\)/@]+\s+){0,3}?(?:in|into|to|per|/)\s*(\d+\.?\d*)\s*(mL|L)\b', t):
        if not _ok(t, m): continue
        mass = f(m.group(1)) * (1 if m.group(2) == 'g' else 1e-3); vol = f(m.group(3)) * (1 if m.group(4) == 'L' else 1e-3)
        if vol > 0 and 0.01 <= mass / vol <= 200:
            put('X__adsorbent_dose_g_L', mass / vol, m); put('X__solution_volume_mL', vol * 1000, m); break
    if 'X__adsorbent_dose_g_L' not in out:
        for m in re.finditer(r'(?:dos(?:e|age)|solid[- ]?(?:to[- ])?liquid(?: ratio)?|S/L(?: ratio)?|adsorbent (?:amount|loading))\D{0,15}?(\d+\.?\d*)\s*(g/L|mg/mL|g L-1|g·L-1|mg/L)', t, re.I):
            if not _ok(t, m): continue
            v = f(m.group(1)); v = v / 1000 if m.group(2) == 'mg/L' else v
            if 0.01 <= v <= 200: put('X__adsorbent_dose_g_L', v, m); break
    for m in re.finditer(r'\bpH\s*(?:=|of|~|≈|:|was|at|value)?\s*(\d{1,2}(?:\.\d+)?)', t):
        if _ok(t, m) and 1 <= f(m.group(1)) <= 14: put('X__initial_pH', f(m.group(1)), m); break
    for m in re.finditer(r'\b(\d+\.?\d*)\s*rpm', t):
        if _ok(t, m) and 50 <= f(m.group(1)) <= 1500: put('X__agitation_rpm', f(m.group(1)), m); break
    for m in re.finditer(r'\b(2[89]\d|3[0-6]\d)(?:\.\d+)?\s*K\b', t):
        if _ok(t, m): put('X__temperature_C', f(m.group(1)) - 273, m); break
    if 'X__temperature_C' not in out:
        for m in re.finditer(r'(?:adsorption|isotherm|kinetic|batch|ads\.?)[^;|]{0,60}?\b([1-8]\d)\s*(?:°\s*C|C)\b', t, re.I):
            if _ok(t, m): put('X__temperature_C', f(m.group(1)), m); break
    if 'X__temperature_C' not in out:
        for m in re.finditer(r'room temperature|ambient temperature', t, re.I):
            if _ok(t, m): put('X__temperature_C', 25.0, m); break
    for m in re.finditer(r'(?:adsorption|contact|equilibr\w*|shak\w*|stirr\w*|agitat\w*)\s*(?:time\s*)?(?:of|for|=|:)?\s*(\d+\.?\d*)\s*(h|hr|hours?|min|minutes?|d|days?)\b', t, re.I):
        if not _ok(t, m): continue
        v = f(m.group(1)); u = m.group(2).lower(); v = v * 60 if u.startswith('h') else (v * 1440 if u.startswith('d') else v)
        if 1 <= v <= 30 * 1440: put('X__contact_time_min', v, m); break
    for pat in [r'(?:Li\+?|lithium)\s*(?:concentration|conc\.?)?\s*(?:=|of|:|~)?\s*(\d+\.?\d*)\s*(mg/L|ppm|mg L-1|g/L|mmol/L|mM)\b',
                r'(\d+\.?\d*)\s*(mg/L|ppm|g/L|mmol/L|mM)\s*(?:of\s*)?(?:Li\+?|lithium|LiCl)\b']:
        for m in re.finditer(pat, t, re.I):
            if not _ok(t, m): continue
            v = f(m.group(1)); u = m.group(2); v = v * 1000 if u == 'g/L' else (v * 6.941 if u in ('mmol/L', 'mM') else v)
            if 0.05 <= v <= 50000: put('X__Li_mg_L', v, m); break
        if 'X__Li_mg_L' in out: break
    return (out, sp) if spans else out

# Manual review (all unique mined snippets were read): snippets below gave a wrong or assumed value and are rejected.
MANUAL_REJECT = [
    ('X__Li_mg_L', 'natural seawater Li 0.17 mg/L, spiked'), ('X__Li_mg_L', 'feed presumed 300 mg/L'),
    ('X__adsorbent_dose_g_L', '10-75 mg adsorbent in 5 L'), ('X__solution_volume_mL', '10-75 mg adsorbent in 5 L'),
    ('X__initial_pH', 'pH 8.5 tris buff'), ('X__initial_pH', 'pH 10 assumed'), ('X__initial_pH', 'when pH lower than 10'),
    ('X__initial_pH', 'at pH<6, 49.75 at pH 13'), ('X__initial_pH', 'optimum pH6'), ('X__initial_pH', 'pH 9 stated op'),
    ('X__initial_pH', '5 mol/L LiCl pH 9 at 95 C'), ('X__initial_pH', '6.05@pH7'),
    ('X__solution_volume_mL', 'assuming per 100 mL'), ('X__adsorbent_dose_g_L', 'assuming per 100 mL'),
    ('X__temperature_C', 'kinetics 25/35/45C'), ('X__temperature_C', 'not 303K batch'), ('X__temperature_C', '25-35 C'),
    ('X__temperature_C', 'Room temperature implied'),
    # V13 new-row review (2026-10-09)
    ('X__adsorbent_dose_g_L', 'pre-unloaded in DI water'), ('X__solution_volume_mL', 'pre-unloaded in DI water'),
    ('X__temperature_C', '303/313/323/333 K'), ('X__temperature_C', 'desorb 150 mL DI water 40 C'),
]

def mine_reviewed(t):
    out, sp = mine(t, True)
    for col, snip in MANUAL_REJECT:
        if col in out and snip.lower() in t.lower() and snip.lower().split()[0] in sp[col].lower():
            out.pop(col)
    return out

COND = ['X__adsorbent_dose_g_L','X__solution_volume_mL','X__initial_pH','X__agitation_rpm','X__temperature_C',
        'X__contact_time_min','X__Li_mg_L']
PAPER_FILL = COND + ['X__calcination_temp_C','X__acid_treatment_conc_M','X__acid_treatment_agent','X__BET_m2_g',
        'X__particle_size_um','X__pore_size_nm','X__pore_volume_cm3_g','X__adsorbent_form','X__mechanism_class']

def complete(df, log=None):
    df = df.copy()
    for c in COND: df[c] = df[c].map(parse_num)
    n_text = {c: 0 for c in COND}
    for i, r in df.iterrows():
        mined = mine_reviewed(txt(r))
        for c, v in mined.items():
            if pd.isna(df.at[i, c]): df.at[i, c] = v; n_text[c] += 1
    # within-paper protocol fill: only when the paper reports exactly ONE distinct value for the column,
    # and only within the same material family + experiment type (same protocol)
    n_paper = {}
    key = [df['meta__paper_id'], df['X__material_family'].fillna('?'), df['X__experiment_type'].fillna('?')]
    for c in PAPER_FILL:
        before = df[c].isna().sum()
        def fill(s):
            u = s.dropna().astype(str).unique()
            if len(u) == 1 and s.isna().any():
                return s.fillna(s.dropna().iloc[0])
            return s
        df[c] = df.groupby(key, group_keys=False)[c].apply(fill).reindex(df.index)
        n_paper[c] = before - df[c].isna().sum()
    if log is not None: log.update({'text_mined': n_text, 'paper_protocol_fill': n_paper})
    return df


# ====== curation & feature assembly ======

def full_features(d):
    F = build(d); G = synth_feats(d)
    F = pd.concat([F, G, precursor_feats(d), series_feats(d)], axis=1)
    # non-cycling experiments are first-use: cycle 1
    F['cycle_number'] = F['cycle_number'].where(F['cycle_number'].notna() | (F['cycling_row'] == 1), 1.0)
    return F
CATS2 = CATS + ['basis','Mn_source','Li_source','Ti_source','oxidant']
CATS[:] = CATS2

