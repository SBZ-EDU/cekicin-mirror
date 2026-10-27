#!/usr/bin/env python3
"""Plate pipeline (fast-alpr YOLOv9-t-384 + CCT-S v2 global OCR, ONNX CPU) on CCPD-2019 100k (HF okita-souji/ccpd2019train).
Ground truth parsed from CCPD file names (bbox, plate indices, tilt, brightness, blur). Resumable CSV.
The global OCR has no Chinese characters, so text is scored on the 6 Latin/digit characters after the province glyph.
  python ccpd100k.py [--limit N] [--threads 1] [--every K]
"""
import argparse, csv, os, re, time, zipfile
import numpy as np, cv2, onnxruntime as ort
from fast_alpr import ALPR
ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int); ap.add_argument('--threads', type=int, default=1); ap.add_argument('--tag', default='ccpd_alpr')
a = ap.parse_args(); OUT = '/home/user/cekicin-ai/v16/runs'; os.makedirs(OUT, exist_ok=True)
PROV = list('皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学') + ['O']
ALPH = list('ABCDEFGHJKLMNPQRSTUVWXYZ') + ['O']
ADS = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789') + ['O']
def parse(n):
    f = os.path.splitext(os.path.basename(n))[0].split('-')
    tilt = [int(x) for x in f[1].split('_')]; (x1, y1), (x2, y2) = [[int(v) for v in p.split('&')] for p in f[2].split('_')]
    idx = [int(x) for x in f[4].split('_')]
    return dict(box=[x1, y1, x2, y2], prov=PROV[idx[0]], gt6=ALPH[idx[1]] + ''.join(ADS[i] for i in idx[2:]), htilt=tilt[0], vtilt=tilt[1], bright=int(f[5]), blur=int(f[6]),
                area=float(f[0]))
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i; return i / u if u else 0
norm = lambda s: re.sub(r'[^A-Z0-9]', '', (s or '').upper())
def lev(x, y):
    dp = list(range(len(y) + 1))
    for i, ca in enumerate(x, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(y, 1): prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1
so2 = ort.SessionOptions(); so2.intra_op_num_threads = a.threads; so2.inter_op_num_threads = 1
alpr = ALPR(detector_model='yolo-v9-t-384-license-plate-end2end', ocr_model='cct-s-v2-global-model', ocr_device='cpu',
            detector_sess_options=so, ocr_sess_options=so2)
z = zipfile.ZipFile('/home/user/data49/ccpd_train.zip'); names = [n for n in z.namelist() if n.endswith('.jpg')]
path = f'{OUT}/{a.tag}.csv'; done = set()
if os.path.exists(path): done = {r['file'] for r in csv.DictReader(open(path))}
new = not done; fh = open(path, 'a', newline=''); w = csv.writer(fh)
if new: w.writerow(['file', 'prov', 'gt6', 'htilt', 'vtilt', 'bright', 'blur', 'box_area', 'n_det', 'det_conf', 'iou', 'pred', 'pred6', 'ok6', 'cer6', 'ocr_conf', 'ms'])
n = len(done); k0 = n; t0 = time.time()
for nm in names:
    if nm in done: continue
    if a.limit and n >= a.limit: break
    g = parse(nm)
    try:
        img = cv2.imdecode(np.frombuffer(z.read(nm), np.uint8), cv2.IMREAD_COLOR)
        ts = time.time(); res = alpr.predict(img); ms = (time.time() - ts) * 1000
    except Exception as e:
        w.writerow([nm, g['prov'], g['gt6'], g['htilt'], g['vtilt'], g['bright'], g['blur'], 0, -1, 0, 0, 'ERR:' + str(e)[:40], '', 0, 1, 0, 0]); n += 1; continue
    best = None
    for r in res:
        b = r.detection.bounding_box; c = float(r.detection.confidence)
        if best is None or c > best[1]: best = (r, c, [b.x1, b.y1, b.x2, b.y2])
    bx = g['box']; barea = (bx[2] - bx[0]) * (bx[3] - bx[1])
    if best:
        r, c, box = best; txt = norm(r.ocr.text if r.ocr else ''); oc = r.ocr.confidence if r.ocr else 0
        oc = float(np.mean(oc)) if isinstance(oc, (list, tuple, np.ndarray)) else float(oc or 0)
        p6 = txt[-6:] if len(txt) >= 6 else txt
        w.writerow([nm, g['prov'], g['gt6'], g['htilt'], g['vtilt'], g['bright'], g['blur'], barea, len(res), round(c, 3), round(iou(box, bx), 3), txt, p6, int(p6 == g['gt6']),
                    round(lev(p6, g['gt6']) / 6, 3), round(oc, 3), round(ms, 1)])
    else:
        w.writerow([nm, g['prov'], g['gt6'], g['htilt'], g['vtilt'], g['bright'], g['blur'], barea, 0, 0, 0, '', '', 0, 1, 0, round(ms, 1)])
    n += 1
    if n % 1000 == 0:
        fh.flush(); el = time.time() - t0; print(f'{n} done  {(n - k0) / el:.1f} img/s', flush=True)
fh.close(); print('FINISHED', n, flush=True)
