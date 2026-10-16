"""
eval_clm_on_splits.py
=====================
FCI CLM'yi modellerimizle AYNI örnekler (split_v1_compact test kümeleri) ve aynı etiketler üzerinde değerlendirir.
Girdi: data/processed/h5meta_train_v2.npz (kompakt H5'ten labels/okta/wmoids/obs_time/timestamps),
       data/processed/clm_station/ (src/extract_fci_clm.py çıktısı), splits/split_v1_compact.npz.
CLM kararları:
  merkez       : cloud_state in {2,3} -> bulutlu, {1,8} -> açık (diğerleri geçersiz)
  alan (gN)    : (bulaşmış+dolu)/geçerli >= eşik; grup ve eşik YALNIZ train örneklerinde seçilir
Kırılımlar: etiket kaynağı (SYNOP = gözlem dakikası :00, METAR = :20/:50), CLM dönemi (C/O), aydınlanma.
Kullanım: python src/eval_clm_on_splits.py  [--out artifacts/clm_on_splits.json]
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

GROUPS = ['c1', '3x3', 'r5', 'r10', 'r20', 'r30']

def metrics(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    rc = tp / max(tp + fn, 1); rr = tn / max(tn + fp, 1)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {'n': len(y), 'balanced_accuracy': 0.5 * (rc + rr), 'recall_cloudy': rc, 'recall_clear': rr,
            'mcc': (tp * tn - fp * fn) / den if den > 0 else float('nan'), 'cloudy_ratio': float(y.mean()) if len(y) else float('nan')}

def load_clm_for(keys, clm_dir):
    """keys: DataFrame [rc_start, wmoid] -> CLM özellikleri (merkez + grup oranları)."""
    keys = keys.assign(day=keys.rc_start.str[:8])
    parts = []
    for day, g in keys.groupby('day'):
        p = os.path.join(clm_dir, f'clm_{day}.npz')
        if not os.path.exists(p):
            continue
        z = np.load(p)
        ri = {r: i for i, r in enumerate(z['rc_start'])}; si = {int(w): i for i, w in enumerate(z['wmoid'])}
        fi = g.rc_start.map(ri); wi = g.wmoid.map(si); ok = fi.notna() & wi.notna()
        if not ok.any():
            continue
        g = g[ok]; fi = fi[ok].astype(int).values; wi = wi[ok].astype(int).values
        c = z['counts'][fi, wi].astype(np.int32)
        valid = np.maximum(c[..., [1, 2, 3, 8]].sum(-1), 1)
        d = {'row': g.index.values, 'disp': z['disposition'][fi], 'cstate': z['center_state'][fi, wi], 'illum': z['center_illum'][fi, wi]}
        for k, gn in enumerate(GROUPS):
            d['f_' + gn] = (c[:, k, 2] + c[:, k, 3]) / valid[:, k]
        parts.append(pd.DataFrame(d))
    return pd.concat(parts).set_index('row')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', default='data/processed/h5meta_train_v2.npz')
    ap.add_argument('--clm_dir', default='data/processed/clm_station')
    ap.add_argument('--split', default='splits/split_v1_compact.npz')
    ap.add_argument('--out', default='artifacts/clm_on_splits.json')
    a = ap.parse_args()
    z = np.load(a.meta, allow_pickle=False)
    meta = pd.DataFrame({'label': z['labels'], 'wmoid': z['wmoids'].astype(int),
                         'src': np.where(pd.to_datetime(z['obs_time'], unit='s').minute == 0, 'SYNOP', 'METAR')})
    meta['rc_start'] = pd.to_datetime(pd.Series(z['timestamps']).str[:14], format='%Y%m%d%H%M%S').dt.floor('10min').dt.strftime('%Y%m%d%H%M').values
    sp = np.load(a.split)
    res = {}
    # 1) eşik ve grup seçimi yalnız train'de
    tr = meta.iloc[sp['train']]; tr = tr[tr.label >= 0]
    ctr = load_clm_for(tr[['rc_start', 'wmoid']], a.clm_dir).join(tr[['label']], how='inner')
    ctr = ctr[ctr.cstate.isin([1, 2, 3, 8])]
    best = {}
    for gn in GROUPS[1:]:
        cand = [(metrics(ctr.label, ctr['f_' + gn] >= th)['balanced_accuracy'], round(float(th), 2)) for th in np.arange(0.05, 1.0, 0.05)]
        best[gn] = max(cand)
    gsel = max(best, key=lambda g: best[g][0])
    res['train_selection'] = {'per_group': {g: {'train_balacc': b[0], 'threshold': b[1]} for g, b in best.items()},
                              'selected_group': gsel, 'n_train': len(ctr)}
    print('train seçimi:', {g: (round(b[0], 4), b[1]) for g, b in best.items()}, '-> seçilen', gsel)
    # 2) test kümeleri
    for name in ('val', 'test_time', 'test_station', 'test_both'):
        s = meta.iloc[sp[name]]; s = s[s.label >= 0]
        c = load_clm_for(s[['rc_start', 'wmoid']], a.clm_dir).join(s[['label', 'src']], how='inner')
        cover = len(c) / len(s)
        c = c[c.cstate.isin([1, 2, 3, 8])]
        out = {'n_subset': len(s), 'clm_coverage': cover, 'n_eval': len(c)}
        def block(df):
            r = {'center': metrics(df.label, df.cstate.isin([2, 3]))}
            for gn in GROUPS[1:]:
                r[gn] = metrics(df.label, df['f_' + gn] >= best[gn][1])
            return r
        out['all'] = block(c)
        for src in ('SYNOP', 'METAR'):
            out['src_' + src] = block(c[c.src == src])
        for d in ('C', 'O'):
            if (c.disp == d).sum() > 200:
                out['disp_' + d] = block(c[c.disp == d])
        for il, nm in ((1, 'night'), (2, 'twilight'), (3, 'day')):
            if (c.illum == il).sum() > 200:
                out['illum_' + nm] = {'center': metrics(c[c.illum == il].label, c[c.illum == il].cstate.isin([2, 3])),
                                      gsel: metrics(c[c.illum == il].label, c[c.illum == il]['f_' + gsel] >= best[gsel][1])}
        res[name] = out
        print(f"\n== {name}: n={len(s)} CLM kapsamı={cover:.3f} değerlendirilen={len(c)}")
        for k in ('all', 'src_SYNOP', 'src_METAR', 'disp_C', 'disp_O'):
            if k in out:
                m0 = out[k]['center']; m1 = out[k][gsel]
                print(f"  {k:10s} n={m0['n']:6d}  merkez BalAcc {m0['balanced_accuracy']:.3f} (R_bul {m0['recall_cloudy']:.3f}, R_açık {m0['recall_clear']:.3f}) | "
                      f"{gsel} BalAcc {m1['balanced_accuracy']:.3f} (R_bul {m1['recall_cloudy']:.3f}, R_açık {m1['recall_clear']:.3f}) MCC {m1['mcc']:.3f}")
        for k in ('illum_night', 'illum_twilight', 'illum_day'):
            if k in out:
                print(f"  {k:14s} merkez {out[k]['center']['balanced_accuracy']:.3f} | {gsel} {out[k][gsel]['balanced_accuracy']:.3f}  (n={out[k]['center']['n']})")
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2, default=float)
    print('\nYazıldı:', a.out)

if __name__ == '__main__':
    main()
