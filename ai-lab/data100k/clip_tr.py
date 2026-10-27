#!/usr/bin/env python3
"""CLIP ViT-B/32 (fp32, brand = aggregated pair probs) vs llama-4-scout VLM on the SAME images: TR-67 (Turkish market) and SC-120 (Stanford Cars).
Candidate label set = DVM 811 UK pairs + all test labels (closed set; disclosed). VLM answers are open-set. Paired exact McNemar."""
import json, math, re, unicodedata, sys, os
import numpy as np, pandas as pd, onnxruntime as ort
sys.argv = ['x']; exec(open('/home/user/cekicin-ai/v16/car100k.py').read().split('def main():')[0])
R46 = '/home/user/research46'
def nz(s): s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower(); s = re.sub(r'[^a-z0-9]', '', s); return {'vauxhall': 'opel', 'mercedesbenz': 'mercedes', 'amgeneral': 'hummer', 'chevy': 'chevrolet'}.get(s, s)
MK = ['AM General', 'Aston Martin', 'Land Rover', 'Mercedes-Benz', 'Rolls-Royce', 'Alfa Romeo', 'Acura', 'Audi', 'BMW', 'Bentley', 'Bugatti', 'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Daewoo', 'Dodge', 'Eagle', 'FIAT', 'Ferrari', 'Fisker', 'Ford', 'GMC', 'Geo', 'HUMMER', 'Honda', 'Hyundai', 'Infiniti', 'Isuzu', 'Jaguar', 'Jeep', 'Lamborghini', 'Lincoln', 'Maybach', 'Mazda', 'McLaren', 'Mitsubishi', 'Nissan', 'Plymouth', 'Porsche', 'Ram', 'Scion', 'Spyker', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo', 'smart']
def split_sc(label):
    lab = re.sub(r'\s(19|20)\d\d$', '', label.strip()); m = max([k for k in MK if lab.lower().startswith(k.lower() + ' ')], key=len, default=lab.split()[0]); return m, lab[len(m):].strip()
def mcn(b, c):
    n = b + c; k = min(b, c); return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [round(100 * p, 1), round(100 * (c - h), 1), round(100 * (c + h), 1)]
df = pd.read_parquet(f'{D}/dvm.parquet'); pairs = sorted({(str(m), str(g)) for m, g in zip(df.Maker, df.Genmodel)}); del df
tr = json.load(open(f'{R46}/results/vlm_tr69_llama4scout.json')); sc = json.load(open(f'{R46}/results/vlm_sc120_llama4scout.json'))
meta = {m['file']: m for m in json.load(open(f'{R46}/trcars/meta_clean.json'))}
sets = {}
uf = set(tr['unique_files']); rows = [r for r in tr['rows'] if r['file'] in uf and r['file'] in meta]
sets['TR-67'] = [(r['file'], meta[r['file']]['make'], meta[r['file']]['model'], r) for r in rows]
sets['SC-120'] = [(r['file'], *split_sc(r['label']), r) for r in sc['rows']]
extra = sorted({(m, g) for s in sets.values() for _, m, g, _ in s}); allp = sorted(set(pairs) | set(extra)); makers = sorted({nz(m) for m, _ in allp})
tok = Tokenizer.from_file(f'{M}/tokenizer.json'); tok.enable_padding(length=77, pad_id=49407); tok.enable_truncation(77)
ts = ort.InferenceSession(f'{M}/text_model.onnx', providers=['CPUExecutionProvider'])
def enc(texts):
    out = []
    for i in range(0, len(texts), 64):
        e = tok.encode_batch(texts[i:i + 64]); out.append(ts.run(['text_embeds'], {'input_ids': np.array([x.ids for x in e], np.int64)})[0])
    v = np.concatenate(out); return v / np.linalg.norm(v, axis=1, keepdims=True)
T = ['a photo of a {}.', 'a photo of a {}, a type of car.', 'a photo of my {}.', 'a close-up photo of a {} car.']
ep = np.mean([enc([t.format(f'{m} {g}') for m, g in allp]) for t in T], axis=0); ep /= np.linalg.norm(ep, axis=1, keepdims=True); del ts
pm = np.array([makers.index(nz(m)) for m, _ in allp])
vs = ort.InferenceSession(f'{M}/vision_model.onnx', providers=['CPUExecutionProvider']); OUT = {}
for name, items in sets.items():
    X = np.stack([prep(open(f'{R46}/{f}', 'rb').read()) for f, _, _, _ in items])
    f = np.concatenate([vs.run(['image_embeds'], {'pixel_values': X[i:i + 8]})[0] for i in range(0, len(X), 8)]); f /= np.linalg.norm(f, axis=1, keepdims=True)
    l = 100 * f @ ep.T; l -= l.max(1, keepdims=True); P = np.exp(l); P /= P.sum(1, keepdims=True)
    cb, cm, vb, vm, ex = [], [], [], [], []
    for j, (fn, mk, md, r) in enumerate(items):
        agg = np.zeros(len(makers)); np.add.at(agg, pm, P[j]); pb = makers[int(agg.argmax())]; pp = allp[int(P[j].argmax())]
        cb.append(pb == nz(mk)); cm.append(nz(pp[0]) == nz(mk) and nz(pp[1]) == nz(md)); vb.append(bool(r['make_ok'])); vm.append(bool(r['model_ok']))
        ex.append(dict(file=fn, gt=f'{mk} {md}', clip=f'{pp[0]} {pp[1]}', clip_brand_p=round(float(agg.max()), 3), vlm=f"{r.get('pred_make')} {r.get('pred_model')}"))
    n = len(items); cb, cm, vb, vm = map(np.array, (cb, cm, vb, vm))
    OUT[name] = dict(n=n, clip_brand=wil(int(cb.sum()), n), vlm_brand=wil(int(vb.sum()), n), p_brand=mcn(int((cb & ~vb).sum()), int((~cb & vb).sum())),
                     clip_model=wil(int(cm.sum()), n), vlm_model=wil(int(vm.sum()), n), p_model=mcn(int((cm & ~vm).sum()), int((~cm & vm).sum())), examples=ex[:12], n_labels=len(allp))
    print(name, {k: v for k, v in OUT[name].items() if k != 'examples'})
json.dump(OUT, open('/home/user/cekicin-ai/v16/runs/clip_vs_vlm.json', 'w'), indent=1, ensure_ascii=False)
