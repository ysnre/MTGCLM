"""
eval_nwc_and_metar.py
=====================
(1) NWC SAF Cloud Type (MSG, MGM PNG) — split_v1_compact örneklerinde ikili bulut skoru (merkez + alan, eşik train'de).
(2) METAR "açık" (NSC/CAVOK) etiketlerinin doğrulanması: aynı istasyonun en yakın saatteki SYNOP N'i ile
    (doğrulanmış N=0 / parçalı 1-4 / bulutlu 5-8 / SYNOP yok) ve bu grupların CLM + NWC ile uyumu.
(3) NWC bulut tipine göre kırılım: METAR-NSC açık etiketleri ve CLM'nin kaçırdığı bulutlu vakalar hangi tipte?
Girdi: data/processed/h5meta_train_v2.npz, data/processed/{clm_station,nwcsaf_station}/, splits/split_v1_compact.npz,
       data/raw_observation/synop_all.csv.   Çıktı: artifacts/nwc_metar_on_splits.json
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_clm_on_splits import metrics, load_clm_for

GROUPS = ['c1', '3x3', 'r5', 'r10', 'r20', 'r30']
NAMES = ["beyaz(açık)", "bulutsuz kara", "bulutsuz deniz", "kar", "deniz buzu", "çok alçak", "alçak", "orta",
         "yüksek opak", "çok yüksek opak", "fraksiyonel", "yük.yarısaydam ince", "yük.yarısaydam orta",
         "yük.yarısaydam kalın", "yük.yarısaydam alçak/orta üstü", "yük.yarısaydam kar üstü"]

def load_nwc_for(keys, nwc_dir):
    keys = keys.assign(day=keys.slot.str[:8]); parts = []
    for day, g in keys.groupby('day'):
        p = os.path.join(nwc_dir, f'nwc_{day}.npz')
        if not os.path.exists(p):
            continue
        z = np.load(p)
        ri = {r: i for i, r in enumerate(z['slot'])}; si = {int(w): i for i, w in enumerate(z['wmoid'])}
        fi = g.slot.map(ri); wi = g.wmoid.map(si); ok = fi.notna() & wi.notna()
        if not ok.any():
            continue
        g = g[ok]; fi = fi[ok].astype(int).values; wi = wi[ok].astype(int).values
        okf = z['ok'][fi]
        c = z['counts'][fi, wi].astype(np.int32)
        tot = np.maximum(c.sum(-1), 1)
        d = {'row': g.index.values[okf], 'ncls': z['center_class'][fi, wi][okf]}
        for k, gn in enumerate(GROUPS):
            d['n_' + gn] = (c[:, k, 5:16].sum(-1) / tot[:, k])[okf]
        parts.append(pd.DataFrame(d))
    return pd.concat(parts).set_index('row')

z = np.load('data/processed/h5meta_train_v2.npz', allow_pickle=False)
ts = pd.to_datetime(pd.Series(z['timestamps']).str[:14], format='%Y%m%d%H%M%S')
ot = pd.to_datetime(z['obs_time'], unit='s')
meta = pd.DataFrame({'label': z['labels'], 'okta': z['okta'], 'wmoid': z['wmoids'].astype(int),
                     'src': np.where(ot.minute == 0, 'SYNOP', 'METAR'),
                     'rc_start': ts.dt.floor('10min').dt.strftime('%Y%m%d%H%M').values,
                     'slot': (ts - pd.Timedelta('9min')).dt.round('15min').dt.strftime('%Y%m%d%H%M').values,
                     'obs_hour': pd.Series(ot).dt.round('h').dt.strftime('%Y%m%d%H').values})
sp = np.load('splits/split_v1_compact.npz')
res = {}

# ---- (2) METAR açık etiketlerini SYNOP ile doğrula
syn = pd.read_csv('data/raw_observation/synop_all.csv', usecols=['validity', 'wmoid', 'cloud_coverage']).dropna()
syn = syn[syn.cloud_coverage >= 0]
syn['wmoid'] = syn.wmoid.astype(int); syn['obs_hour'] = pd.to_datetime(syn.validity).dt.strftime('%Y%m%d%H')
syn = syn.drop_duplicates(['wmoid', 'obs_hour']).set_index(['wmoid', 'obs_hour']).cloud_coverage
key = pd.MultiIndex.from_arrays([meta.wmoid, meta.obs_hour])
meta['synop_N'] = syn.reindex(key).values
mc = (meta.src == 'METAR') & (meta.label == 0)
cat = np.select([meta.synop_N == 0, meta.synop_N.between(1, 4), meta.synop_N.between(5, 9)], ['dogrulandi_N0', 'parcali_N1_4', 'bulutlu_N5_9'], 'synop_yok')
meta['nsc_cat'] = np.where(mc, cat, '')
res['metar_clear_verification'] = {}
for name in ('train', 'val', 'test_time', 'test_station', 'test_both'):
    s = meta.iloc[sp[name]]; m = s[s.nsc_cat != '']
    res['metar_clear_verification'][name] = {'n_metar_clear': int(len(m)), **{k: round(float(v), 4) for k, v in m.nsc_cat.value_counts(normalize=True).items()}}
    print(f"{name:12s} METAR-açık n={len(m):6d}  " + "  ".join(f"{k}={v:.3f}" for k, v in m.nsc_cat.value_counts(normalize=True).items()))

# ---- (1) NWC skor, eşik train'de
tr = meta.iloc[sp['train']]; tr = tr[tr.label >= 0]
ntr = load_nwc_for(tr[['slot', 'wmoid']], 'data/processed/nwcsaf_station').join(tr[['label']], how='inner')
best = {gn: max((metrics(ntr.label, ntr['n_' + gn] >= th)['balanced_accuracy'], round(float(th), 2)) for th in np.arange(0.05, 1.0, 0.05)) for gn in GROUPS[1:]}
gsel = max(best, key=lambda g: best[g][0])
res['nwc_train_selection'] = {g: {'balacc': b[0], 'threshold': b[1]} for g, b in best.items()}; res['nwc_selected_group'] = gsel
print('\nNWC train seçimi:', {g: (round(b[0], 3), b[1]) for g, b in best.items()}, '->', gsel)
for name in ('val', 'test_time', 'test_station', 'test_both'):
    s = meta.iloc[sp[name]]; s = s[s.label >= 0]
    n = load_nwc_for(s[['slot', 'wmoid']], 'data/processed/nwcsaf_station').join(s[['label', 'src', 'nsc_cat']], how='inner')
    out = {'coverage': len(n) / len(s)}
    for sub, df in (('all', n), ('SYNOP', n[n.src == 'SYNOP']), ('METAR', n[n.src == 'METAR'])):
        out[sub] = {'center': metrics(df.label, (df.ncls >= 5) & (df.ncls <= 15)), gsel: metrics(df.label, df['n_' + gsel] >= best[gsel][1])}
    res['nwc_' + name] = out
    print(f"NWC {name:12s} kapsam={out['coverage']:.3f}  " + "  ".join(f"{sub}: merkez {out[sub]['center']['balanced_accuracy']:.3f} / {gsel} {out[sub][gsel]['balanced_accuracy']:.3f}" for sub in ('all', 'SYNOP', 'METAR')))

# ---- (2b) CLM ve NWC'nin METAR-açık alt gruplarında "bulutlu" deme oranı (test_time + val)
for name in ('val', 'test_time'):
    s = meta.iloc[sp[name]]; s = s[(s.label == 0)]
    c = load_clm_for(s[['rc_start', 'wmoid']], 'data/processed/clm_station').join(s[['src', 'nsc_cat']], how='inner')
    c = c[c.cstate.isin([1, 2, 3, 8])]
    n = load_nwc_for(s[['slot', 'wmoid']], 'data/processed/nwcsaf_station').join(s[['src', 'nsc_cat']], how='inner')
    grp = lambda df: np.where(df.src == 'SYNOP', 'SYNOP_N0', df.nsc_cat)
    cc = c.assign(g=grp(c)).groupby('g').apply(lambda d: pd.Series({'n': len(d), 'clm_bulutlu_merkez': d.cstate.isin([2, 3]).mean(), 'clm_bulutlu_r30': (d.f_r30 >= 0.45).mean()}))
    nn = n.assign(g=grp(n)).groupby('g').apply(lambda d: pd.Series({'nwc_bulutlu_merkez': ((d.ncls >= 5) & (d.ncls <= 15)).mean()}))
    tab = cc.join(nn); res['clear_label_agreement_' + name] = tab.round(4).to_dict(orient='index')
    print(f"\nAçık etiketlerde referansların 'bulutlu' deme oranı — {name}\n" + tab.round(3).to_string())

# ---- (3) NWC bulut tipi kırılımı (test_time + val birlikte)
idx = np.concatenate([sp['val'], sp['test_time']]); s = meta.iloc[idx]; s = s[s.label >= 0]
n = load_nwc_for(s[['slot', 'wmoid']], 'data/processed/nwcsaf_station').join(s[['label', 'src', 'nsc_cat', 'rc_start', 'wmoid']], how='inner')
c = load_clm_for(s[['rc_start', 'wmoid']], 'data/processed/clm_station')
n = n.join(c[['cstate']], how='inner')
def dist(mask):
    v = n.ncls[mask].value_counts(normalize=True)
    return {NAMES[int(k)] if k < 16 else 'tanınmayan': round(float(p), 3) for k, p in v.head(6).items()}
res['nwc_type'] = {
    'METAR_NSC_but_SYNOP_cloudy': dist(n.nsc_cat == 'bulutlu_N5_9'),
    'METAR_NSC_verified_clear': dist(n.nsc_cat == 'dogrulandi_N0'),
    'cloudy_label_CLM_center_clear(miss)': dist((n.label == 1) & (n.cstate == 1)),
    'clear_SYNOP_label_CLM_center_cloudy(false_alarm)': dist((n.label == 0) & (n.src == 'SYNOP') & n.cstate.isin([2, 3])),
}
print('\nNWC bulut tipi dağılımı (val + test_time):')
for k, v in res['nwc_type'].items():
    print(f"  {k}: {v}")
json.dump(res, open('artifacts/nwc_metar_on_splits.json', 'w'), indent=2, default=float, ensure_ascii=False)
print('\nYazıldı: artifacts/nwc_metar_on_splits.json')
