#!/usr/bin/env python3
"""Extra experiments for the Vehicle-AI paper: CIs, McNemar, ablation, plate robustness, VLM calibration, error taxonomy."""
import json, os, re, math, time, random, collections, sys
import numpy as np, cv2
R = '/home/user/research46'; OUT = '/home/user/cekicin-ai/v15/exp'; os.makedirs(OUT, exist_ok=True)
J = lambda p: json.load(open(p))
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (p, max(0, c-h), min(1, c+h))
def mcnemar(a, b):
    """exact two-sided McNemar on paired booleans"""
    b01 = sum((not x) and y for x, y in zip(a, b)); b10 = sum(x and (not y) for x, y in zip(a, b)); n = b01 + b10
    if n == 0: return b01, b10, 1.0
    k = min(b01, b10); p = sum(math.comb(n, i) for i in range(0, k+1)) / 2**n * 2
    return b01, b10, min(1.0, p)
res = {}
# ---------- E1/E2 plates: CIs, McNemar raw vs TR-fix, ablation cct-xs vs cct-s
S = J(f'{R}/results/plates_tr250_cct-s-v2-global-model.json'); X = J(f'{R}/results/plates_tr250_cct-xs-v2-global-model.json')
rs = {r['file']: r for r in S['rows']}; rx = {r['file']: r for r in X['rows']}; files = sorted(rs)
n = len(files); raw = [rs[f]['ok_raw'] for f in files]; fix = [rs[f]['ok_fix'] for f in files]; xs = [rx[f]['ok_fix'] for f in files]
res['plates'] = {'n': n, 'exact_raw': wilson(sum(raw), n), 'exact_fix': wilson(sum(fix), n), 'xs_fix': wilson(sum(xs), n),
                 'mcnemar_raw_vs_fix': mcnemar(raw, fix), 'mcnemar_xs_vs_s': mcnemar(xs, fix),
                 'ms_s': S['summary']['ms_median'], 'ms_xs': X['summary'].get('ms_median')}
# bootstrap CI for character accuracy (1 - CER, length-weighted)
cer = np.array([rs[f]['cer'] for f in files]); L = np.array([len(rs[f]['gt']) for f in files]); rng = np.random.default_rng(0)
err = cer * L; bs = []
for _ in range(5000):
    i = rng.integers(0, n, n); bs.append(1 - err[i].sum() / L[i].sum())
res['plates']['char_acc'] = (float(1 - err.sum()/L.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
# error taxonomy: substitution pairs
def align_subs(g, p):
    if len(g) != len(p): return [('len', f'{len(g)}→{len(p)}')]
    return [(a, b) for a, b in zip(g, p) if a != b]
subs = collections.Counter()
for f in files:
    r = rs[f]
    if not r['ok_fix']:
        for a, b in align_subs(r['gt'], r['pred_fix']): subs[f'{a}→{b}'] += 1
res['plates']['confusions'] = subs.most_common(12)
res['plates']['n_err'] = n - sum(fix)
# ---------- E3 robustness of plates (fast-alpr, same models)
if '--skip-robust' not in sys.argv:
    from fast_alpr import ALPR
    sys.path.insert(0, '/home/user/cekicin-ai/v14'); 
    TR_RE = re.compile(r'^(0[1-9]|[1-7][0-9]|8[01])([A-Z]{1,3})([0-9]{2,4})$')
    L2D = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7', 'A': '4'}; D2L = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '7': 'T', '4': 'A'}
    norm = lambda s: re.sub(r'[^A-Z0-9]', '', (s or '').upper().replace('İ', 'I'))
    def tr_fix(s):
        s = norm(s)
        if TR_RE.match(s): return s
        for nl in (1, 2, 3):
            for nd in (2, 3, 4):
                if 2+nl+nd != len(s): continue
                c = ''.join(L2D.get(x, x) for x in s[:2]) + ''.join(D2L.get(x, x) for x in s[2:2+nl]) + ''.join(L2D.get(x, x) for x in s[2+nl:])
                if TR_RE.match(c): return c
        return s
    alpr = ALPR(detector_model='yolo-v9-t-384-license-plate-end2end', ocr_model='cct-s-v2-global-model')
    D = f'{R}/tr250/Turkish_Licence_Plates_250/images_with_texts'
    def deg(img, kind):
        if kind == 'clean': return img
        if kind == 'blur_s2': return cv2.GaussianBlur(img, (0, 0), 2)
        if kind == 'down_0.5': h, w = img.shape[:2]; return cv2.resize(cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA), (w, h))
        if kind == 'down_0.25': h, w = img.shape[:2]; return cv2.resize(cv2.resize(img, (w//4, h//4), interpolation=cv2.INTER_AREA), (w, h))
        if kind == 'jpeg_q15': return cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 15])[1], 1)
        if kind == 'dark_0.35': return np.clip(img.astype(np.float32)*0.35, 0, 255).astype(np.uint8)
        if kind == 'noise_20': return np.clip(img.astype(np.float32) + np.random.default_rng(1).normal(0, 20, img.shape), 0, 255).astype(np.uint8)
        if kind == 'rot_10': h, w = img.shape[:2]; M = cv2.getRotationMatrix2D((w/2, h/2), 10, 1); return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    KINDS = ['clean', 'blur_s2', 'down_0.5', 'down_0.25', 'jpeg_q15', 'dark_0.35', 'noise_20', 'rot_10']
    rob = {}
    imgs = [(f, cv2.imread(os.path.join(D, f))) for f in files]
    for k in KINDS:
        ok = det = 0; t0 = time.time()
        for f, im in imgs:
            gt = norm(re.split(r'[_#]', os.path.splitext(f)[0])[0]); out = alpr.predict(deg(im, k))
            if out: det += 1
            best = max(out, key=lambda r: r.detection.confidence) if out else None
            txt = best.ocr.text if best and best.ocr else ''
            txt = txt[0] if isinstance(txt, list) else txt
            ok += tr_fix(txt) == gt
        rob[k] = {'exact': wilson(ok, n), 'detected': det / n, 'sec': round(time.time()-t0, 1)}
        print(k, rob[k], flush=True)
    res['robustness'] = rob
# ---------- E5 VLM calibration + per-brand
def vlm(path, uniq=None):
    d = J(path); rows = [r for r in d['rows'] if (uniq is None or r['file'] in uniq)]
    m = [r['make_ok'] for r in rows]; mo = [r['model_ok'] for r in rows]; c = [r['conf'] if r['conf'] is not None else 0 for r in rows]
    out = {'n': len(rows), 'make': wilson(sum(m), len(rows)), 'model': wilson(sum(mo), len(rows))}
    # selective accuracy (model level) at confidence thresholds
    sel = []
    for th in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]:
        k = [(x, y) for x, y in zip(c, mo) if x >= th]
        sel.append({'th': th, 'coverage': len(k)/len(rows), 'model_acc': (sum(y for _, y in k)/len(k)) if k else None})
    out['selective'] = sel
    # ECE (model-level correctness, 5 bins)
    bins = np.linspace(0, 1, 6); ece = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = [(x, y) for x, y in zip(c, mo) if lo < x <= hi or (lo == 0 and x == 0)]
        if k: ece += len(k)/len(rows) * abs(np.mean([x for x, _ in k]) - np.mean([y for _, y in k]))
    out['ece_model'] = float(ece); out['mean_conf'] = float(np.mean(c))
    # AUROC conf -> model correct
    pos = [x for x, y in zip(c, mo) if y]; neg = [x for x, y in zip(c, mo) if not y]
    out['auroc_model'] = float(np.mean([(p > q) + 0.5*(p == q) for p in pos for q in neg])) if pos and neg else None
    return out, rows
tr = J(f'{R}/results/vlm_tr69_llama4scout.json')
res['vlm_tr'], tr_rows = vlm(f'{R}/results/vlm_tr69_llama4scout.json', set(tr['unique_files']))
res['vlm_sc'], sc_rows = vlm(f'{R}/results/vlm_sc120_llama4scout.json')
# paired comparison on the subset both models answered (TR + SC), noting it is quota-truncated
for tag, a, b in [('tr', 'vlm_tr69_llama4scout', 'vlm_tr69_mistral31'), ('sc', 'vlm_sc120_llama4scout', 'vlm_sc120_mistral31')]:
    A = {r['file']: r for r in J(f'{R}/results/{a}.json')['rows']}; B = {r['file']: r for r in J(f'{R}/results/{b}.json')['rows'] if r['pred_make'] is not None}
    common = sorted(set(A) & set(B))
    res[f'paired_{tag}'] = {'n': len(common), 'llama_make': sum(A[f]['make_ok'] for f in common)/max(1, len(common)), 'mistral_make': sum(B[f]['make_ok'] for f in common)/max(1, len(common)),
                            'llama_model': sum(A[f]['model_ok'] for f in common)/max(1, len(common)), 'mistral_model': sum(B[f]['model_ok'] for f in common)/max(1, len(common)),
                            'mcnemar_make': mcnemar([A[f]['make_ok'] for f in common], [B[f]['make_ok'] for f in common])}
# ---------- E7 damage street test
res['damage'] = {'flag_precision_strict': wilson(6, 35), 'flag_precision_lenient': wilson(10, 35), 'false_alarm_per_photo': wilson(25, 240),
                 'test1000': {'f1': 0.595, 'recall': 0.840, 'false_alarm': 0.170}}
json.dump(res, open(f'{OUT}/paper_results.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != 'robustness'}, default=float)[:3000])
