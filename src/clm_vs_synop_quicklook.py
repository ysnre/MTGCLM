"""
clm_vs_synop_quicklook.py — ÖN BAKIŞ (12 Eki 2026)
FCI CLM istasyon sayımlarını (src/extract_fci_clm.py çıktısı) ham SYNOP toplam bulutluluğuyla
(N=0 -> açık, N=5..8 -> bulutlu; N=1..4 ve 9 dışarıda) karşılaştırır. Merkez piksel ve alan bazlı
(3x3, r5..r30 km) CLM bulut oranı için BalAcc/AUC; gece/alacakaranlık/gündüz kırılımı.
Zaman eşlemesi: SYNOP HH:00 -> CLM tekrar döngüsü HH-1:50 (Türkiye ~HH-1:58 taranır).
NOT: eşikler aynı veride seçiliyor (iyimser); model karşılaştırması için aynı test örneklerinde
ve eşik yalnız train örneklerinde seçilerek yapılmalıdır.
Kullanım: python src/clm_vs_synop_quicklook.py  (proje kökünden)
"""
import glob, numpy as np, pandas as pd
G = ['c1','3x3','r5','r10','r20','r30']
s = pd.read_csv('data/raw_observation/synop_all.csv', usecols=['validity','wmoid','cloud_coverage']).dropna()
s['wmoid'] = s.wmoid.astype(np.int64)
s = s[(s.cloud_coverage == 0) | s.cloud_coverage.between(5, 8)].copy()
s['y'] = (s.cloud_coverage >= 5).astype(np.int8)
t = pd.to_datetime(s.validity)
s['rc'] = (t - pd.Timedelta('10min')).dt.floor('10min').dt.strftime('%Y%m%d%H%M')
s['day'] = s.rc.str[:8]
sd = {d: g for d, g in s.groupby('day')}
out = {k: [] for k in ['y','disp','cstate','illum'] + ['fi_'+g for g in G] + ['fo_'+g for g in G]}
for f in sorted(glob.glob('data/processed/clm_station/clm_*.npz')):
    day = f[-12:-4]
    if day not in sd: continue
    z = np.load(f); g = sd[day]
    ri = {r: i for i, r in enumerate(z['rc_start'])}; si = {w: i for i, w in enumerate(z['wmoid'])}
    fi = g.rc.map(ri); wi = g.wmoid.map(si); ok = fi.notna() & wi.notna()
    if not ok.any(): continue
    fi = fi[ok].astype(int).values; wi = wi[ok].astype(int).values; gg = g[ok]
    c = z['counts'][fi, wi].astype(np.int32)      # n,G,10
    valid = np.maximum(c[..., [1,2,3,8]].sum(-1), 1)
    out['y'].append(gg.y.values); out['disp'].append(z['disposition'][fi]); out['cstate'].append(z['center_state'][fi, wi]); out['illum'].append(z['center_illum'][fi, wi])
    for k, gname in enumerate(G):
        out['fi_'+gname].append((c[:, k, 2] + c[:, k, 3]) / valid[:, k]); out['fo_'+gname].append(c[:, k, 3] / valid[:, k])
m = pd.DataFrame({k: np.concatenate(v) for k, v in out.items()})
m = m[m.cstate.isin([1,2,3,8])]
print('eşleşen SYNOP–CLM örneği:', len(m), ' bulutlu oranı: %.3f' % m.y.mean())
def ba(y, p):
    tp = ((p==1)&(y==1)).sum(); tn = ((p==0)&(y==0)).sum(); fp = ((p==1)&(y==0)).sum(); fn = ((p==0)&(y==1)).sum()
    return 0.5*(tp/max(tp+fn,1) + tn/max(tn+fp,1)), tp/max(tp+fn,1), tn/max(tn+fp,1)
def auc(y, sc):
    r = pd.Series(sc).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return (r[y==1].sum() - n1*(n1+1)/2) / (n1*n0)
for disp in ('C','O'):
    d = m[m.disp == disp]; y = d.y.values.astype(int)
    print(f'\n=== {disp}  n={len(d)}  bulutlu={y.mean():.3f}')
    print('merkez, bulaşmış=bulutlu : BalAcc %.3f  R(bulutlu) %.3f  R(açık) %.3f' % ba(y, d.cstate.isin([2,3]).astype(int).values))
    print('merkez, yalnız dolu      : BalAcc %.3f  R(bulutlu) %.3f  R(açık) %.3f' % ba(y, (d.cstate==3).astype(int).values))
    for g in G:
        best = max((ba(y, (d['fi_'+g].values >= th).astype(int))[0], th) for th in np.arange(0.05, 1.0, 0.05))
        print(f'{g:4s}: AUC(bulaşmış+dolu oranı) {auc(y, d["fi_"+g].values):.3f}  AUC(dolu oranı) {auc(y, d["fo_"+g].values):.3f}  en iyi eşik BalAcc {best[0]:.3f} @ {best[1]:.2f}')
    for il, nm in ((1,'gece'),(2,'alacakaranlık'),(3,'gündüz')):
        dd = d[d.illum == il]
        if len(dd) > 500:
            print(f'  {nm:13s} n={len(dd):6d}  merkez BalAcc %.3f' % ba(dd.y.values.astype(int), dd.cstate.isin([2,3]).astype(int).values)[0])
