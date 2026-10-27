#!/usr/bin/env python3
"""Analyse the 100k runs (CCPD plates, DVM-Car make/model/body) -> runs/summary49.json + figures fig49/*.png"""
import json, math, os, re, collections
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
V = '/home/user/cekicin-ai/v16'; R = f'{V}/runs'; F = f'{V}/fig49'; os.makedirs(F, exist_ok=True)
def wil(k, n, z=1.96):
    if n == 0: return [0, 0, 0]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * p, 2), round(100 * (c - h), 2), round(100 * (c + h), 2)]
def ece(conf, ok, bins=10):
    conf, ok = np.asarray(conf, float), np.asarray(ok, float); e = 0
    for i in range(bins):
        m = (conf > i / bins) & (conf <= (i + 1) / bins)
        if m.any(): e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return round(float(e), 3)
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n; return min(1.0, 2 * p)
S = {}
# ------------------------------------------------ CCPD plates
d = pd.read_csv(f'{R}/ccpd_alpr.csv', dtype={'gt6': str, 'pred': str, 'pred6': str}); d['pred'] = d.pred.fillna(''); d['pred6'] = d.pred6.fillna('')
n = len(d); det = (d.iou >= 0.5); anyd = d.n_det > 0
P = dict(n=n, det_iou50=wil(int(det.sum()), n), any_det=wil(int(anyd.sum()), n), exact6=wil(int(d.ok6.sum()), n),
         exact6_given_det=wil(int(d.ok6[det].sum()), int(det.sum())), char_acc=round(100 * (1 - d.cer6.clip(0, 1).mean()), 2),
         ms_median=float(d.ms.median()), ms_p90=float(d.ms.quantile(.9)), mean_iou=round(float(d.iou[anyd].mean()), 3))
def cnfix(p):
    # CCPD format: pos0 = letter (no I/O), pos1-5 = letters (no I/O) or digits  -> I->1, O->0 everywhere after pos0; digit look-alikes at pos0 -> letters
    if len(p) != 6: return p
    D2L = {'4': 'A', '8': 'B', '0': 'D', '6': 'G', '2': 'Z', '5': 'S'}
    return D2L.get(p[0], p[0]) + p[1:].replace('I', '1').replace('O', '0')
d['pred6_fix'] = d.pred6.map(cnfix); d['ok6_fix'] = (d.pred6_fix == d.gt6).astype(int)
P['exact6_cnfix'] = wil(int(d.ok6_fix.sum()), n); b1 = int(((d.ok6_fix == 1) & (d.ok6 == 0)).sum()); b2 = int(((d.ok6_fix == 0) & (d.ok6 == 1)).sum())
P['mcnemar_cnfix'] = dict(b=b1, c=b2, p=mcnemar(b1, b2))
P['len_dist'] = {str(k): int(v) for k, v in d.pred.str.len().value_counts().sort_index().items()}
P['ocr_ece'] = ece(d.ocr_conf[det], d.ok6[det])
# selective prediction: OCR conf threshold
P['selective'] = []
for t in [0, .5, .7, .8, .9, .95]:
    m = det & (d.ocr_conf >= t); P['selective'].append(dict(thr=t, coverage=round(100 * m.mean(), 1), acc=wil(int(d.ok6[m].sum()), int(m.sum()))[0]))
# by blur / brightness / tilt / plate size (quantile bins)
def by(col, q=5, f=None):
    x = d[col] if f is None else f(d); b = pd.qcut(x.rank(method='first'), q, labels=False); out = []
    for i in range(q):
        m = b == i; out.append(dict(lo=float(x[m].min()), hi=float(x[m].max()), n=int(m.sum()), det=round(100 * det[m].mean(), 1), exact6=wil(int(d.ok6[m].sum()), int(m.sum()))))
    return out
P['by_blur'] = by('blur'); P['by_bright'] = by('bright'); P['by_area'] = by('box_area'); P['by_tilt'] = by('tilt', f=lambda d: (d.htilt - 90).abs() + (d.vtilt - 90).abs())
P['by_prov'] = {k: dict(n=int(len(g)), exact6=round(100 * g.ok6.mean(), 1)) for k, g in d.groupby('prov') if len(g) >= 200}
# character confusions (same-length aligned)
cc = collections.Counter(); pos = collections.Counter(); posn = collections.Counter()
for g, p in zip(d.gt6[det], d.pred6[det]):
    if len(p) == 6:
        for i, (a, b) in enumerate(zip(g, p)):
            posn[i] += 1
            if a != b: cc[(a, b)] += 1; pos[i] += 1
P['top_confusions'] = [[f'{a}->{b}', k] for (a, b), k in cc.most_common(15)]
P['err_by_position'] = {i: round(100 * pos[i] / posn[i], 2) for i in range(6) if posn[i]}
P['examples_fail'] = d[det & (d.ok6 == 0)].head(12)[['file', 'gt6', 'pred', 'blur', 'bright']].to_dict('records')
S['ccpd'] = P
# ------------------------------------------------ DVM cars
c = pd.read_csv(f'{R}/dvm_clip.csv'); c = c[c.p_maker != 'ERR']; nc = len(c)
c['pm_pair'] = c.p_pair.str.split('|').str[0]; c['gt_pair'] = c.maker + '|' + c.genmodel.astype(str)
ok_m = c.maker == c.p_maker; ok_mp = c.maker == c.pm_pair; ok_ma = c.maker == c.p_maker_agg; ok_p = c.gt_pair == c.p_pair
ok_t5 = c.apply(lambda r: r.maker in str(r.top5_makers).split('|'), axis=1)
cb = c[c.body.notna()]; ok_b = cb.body == cb.p_body
C = dict(n=nc, makers=int(c.maker.nunique()), pairs=int(c.gt_pair.nunique()), maker_prompt=wil(int(ok_m.sum()), nc), maker_from_pair=wil(int(ok_mp.sum()), nc),
         maker_agg=wil(int(ok_ma.sum()), nc), maker_top5_prompt=wil(int(ok_t5.sum()), nc), model=wil(int(ok_p.sum()), nc), body=wil(int(ok_b.sum()), len(cb)), n_body=len(cb),
         ms_median=float(c.ms.median()))
b1 = int((ok_ma & ~ok_m).sum()); b2 = int((~ok_ma & ok_m).sum()); C['mcnemar_agg_vs_prompt'] = dict(b=b1, c=b2, p=mcnemar(b1, b2))
C['ece_maker_agg'] = ece(c.p_maker_agg_prob, ok_ma); C['ece_body'] = ece(cb.p_body_prob, ok_b); C['mean_conf_agg'] = round(float(c.p_maker_agg_prob.mean()), 3)
C['selective_maker'] = []
for t in [0, .3, .5, .7, .9]:
    m = c.p_maker_agg_prob >= t; C['selective_maker'].append(dict(thr=t, coverage=round(100 * m.mean(), 1), acc=wil(int(ok_ma[m].sum()), int(m.sum()))[0]))
VIEW = {0: 'front', 45: 'front-side 45', 90: 'side 90', 135: 'rear-side 135', 180: 'rear', 225: 'rear-side 225', 270: 'side 270', 315: 'front-side 315', 360: 'front 360'}
C['by_view'] = {VIEW.get(int(k), str(k)): dict(n=int(len(g)), maker=round(100 * (g.maker == g.p_maker_agg).mean(), 1), model=round(100 * (g.gt_pair == g.p_pair).mean(), 1)) for k, g in c.groupby('view')}
TRB = ['Renault', 'Fiat', 'Volkswagen', 'Ford', 'Toyota', 'Hyundai', 'Peugeot', 'Vauxhall', 'Dacia', 'Citroen', 'Honda', 'SKODA', 'SEAT', 'Kia', 'BMW', 'Mercedes-Benz', 'Audi', 'Nissan', 'Volvo', 'Mazda']
C['by_maker'] = {m: dict(n=int(len(g)), maker=round(100 * (g.maker == g.p_maker_agg).mean(), 1), model=round(100 * (g.gt_pair == g.p_pair).mean(), 1)) for m, g in c.groupby('maker') if len(g) >= 150}
C['tr_market_makers'] = {m: C['by_maker'][m] for m in TRB if m in C['by_maker']}
C['top_maker_confusions'] = [[f'{a} -> {b}', int(k)] for (a, b), k in collections.Counter(zip(c.maker[~ok_ma], c.p_maker_agg[~ok_ma])).most_common(12)]
labs = sorted(cb.body.unique(), key=lambda x: -(cb.body == x).sum()); cm = pd.crosstab(cb.body, cb.p_body).reindex(index=labs, columns=labs, fill_value=0)
C['body_per_class'] = {k: dict(n=int((cb.body == k).sum()), recall=round(100 * cm.loc[k, k] / max(1, cm.loc[k].sum()), 1)) for k in labs}
C['by_year'] = {f'{int(lo)}-{int(lo)+4}': round(100 * (g.maker == g.p_maker_agg).mean(), 1) for lo, g in c.groupby((c.year // 5) * 5) if len(g) >= 300 and not math.isnan(lo)}
S['dvm'] = C
if os.path.exists(f'{R}/clip_vs_vlm.json'): S['clip_vs_vlm'] = json.load(open(f'{R}/clip_vs_vlm.json'))
json.dump(S, open(f'{R}/summary49.json', 'w'), indent=1, ensure_ascii=False)
# ------------------------------------------------ figures
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, key, lab in [(ax[0], 'by_blur', 'CCPD blur index (low = blurrier)'), (ax[1], 'by_area', 'plate box area (px², quintiles)'), (ax[2], 'by_tilt', 'tilt |Δ| (deg, quintiles)')]:
    rows = P[key]; xs = [f"{r['lo']:.0f}–{r['hi']:.0f}" for r in rows]; y = [r['exact6'][0] for r in rows]; e = [[r['exact6'][0] - r['exact6'][1] for r in rows], [r['exact6'][2] - r['exact6'][0] for r in rows]]
    a.bar(range(5), y, yerr=e, color='#0f3a6b', capsize=3); a.plot(range(5), [r['det'] for r in rows], 'o-', color='#e8590c', label='detection')
    a.set_xticks(range(5)); a.set_xticklabels(xs, fontsize=7.5, rotation=15); a.set_ylim(0, 105); a.set_title(lab, fontsize=9.5); a.set_ylabel('%')
ax[0].legend(['detection (IoU≥0.5)', 'exact 6-char read'], fontsize=8, loc='lower right')
fig.suptitle(f'CCPD-2019, n = {n:,} real photos: Çekicin plate pipeline (no Chinese training)', fontsize=11); fig.tight_layout(); fig.savefig(f'{F}/ccpd_bins.png', dpi=150); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
vs = list(C['by_view']); ax[0].bar(range(len(vs)), [C['by_view'][v]['maker'] for v in vs], color='#0f3a6b', label='brand'); ax[0].bar(range(len(vs)), [C['by_view'][v]['model'] for v in vs], color='#e8590c', width=.5, label='model')
ax[0].set_xticks(range(len(vs))); ax[0].set_xticklabels(vs, rotation=30, fontsize=7.5, ha='right'); ax[0].set_ylabel('%'); ax[0].legend(fontsize=8); ax[0].set_title('by viewpoint', fontsize=10)
tm = C['tr_market_makers']; ks = sorted(tm, key=lambda k: -tm[k]['maker']); ax[1].barh(range(len(ks)), [tm[k]['maker'] for k in ks], color='#0f766e'); ax[1].set_yticks(range(len(ks))); ax[1].set_yticklabels(ks, fontsize=8); ax[1].invert_yaxis()
ax[1].set_xlabel('brand accuracy %'); ax[1].set_title('brands common in Türkiye (Vauxhall = Opel)', fontsize=10)
fig.suptitle(f'DVM-Car, n = {nc:,}: CLIP ViT-B/32 zero-shot brand/model', fontsize=11); fig.tight_layout(); fig.savefig(f'{F}/dvm_view_maker.png', dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(5.6, 4.6)); cmn = cm.div(cm.sum(1), axis=0).values * 100
ax.imshow(cmn, cmap='Blues', vmin=0, vmax=100); ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs))); ax.set_xticklabels(labs, rotation=40, ha='right', fontsize=8); ax.set_yticklabels(labs, fontsize=8)
for i in range(len(labs)):
    for j in range(len(labs)):
        if cmn[i, j] >= 5: ax.text(j, i, f'{cmn[i, j]:.0f}', ha='center', va='center', fontsize=7, color='white' if cmn[i, j] > 50 else 'black')
ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(f'Body type, n = {len(cb):,} (row %)', fontsize=10); fig.tight_layout(); fig.savefig(f'{F}/dvm_body_cm.png', dpi=150); plt.close()
print(json.dumps({k: S['ccpd'][k] for k in ['n', 'det_iou50', 'exact6', 'exact6_cnfix', 'mcnemar_cnfix', 'char_acc', 'ocr_ece']}), '\n', json.dumps({k: S['dvm'][k] for k in ['n', 'maker_prompt', 'maker_from_pair', 'maker_agg', 'model', 'body', 'ece_maker_agg', 'mcnemar_agg_vs_prompt']}))
