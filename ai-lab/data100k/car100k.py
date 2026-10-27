#!/usr/bin/env python3
"""Zero-shot CLIP (ViT-B/32 ONNX, CPU) make / model / body-type test on DVM-Car 100k (HF multabench/extended-img-reg-dvm-car).
Resumable: appends to runs/dvm_clip.csv; rerun skips done rows.
  python car100k.py [--limit N] [--model quantized|fp32] [--threads 1]
"""
import argparse, csv, io, json, os, re, sys, time, zipfile, glob
import numpy as np, pandas as pd, onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer
D = '/home/user/data49'; M = '/home/user/models49/clip'; OUT = '/home/user/cekicin-ai/v16/runs'
ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int); ap.add_argument('--model', default='fp32'); ap.add_argument('--threads', type=int, default=1)
ap.add_argument('--tag', default='dvm_clip'); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
BODY = {'Hatchback': 'hatchback', 'SUV': 'SUV', 'MPV': 'minivan (MPV)', 'Saloon': 'sedan', 'Coupe': 'coupe', 'Estate': 'station wagon', 'Convertible': 'convertible',
        'Pickup': 'pickup truck', 'Van': 'van'}
VAN = {'Combi Van', 'Panel Van', 'Minibus', 'Car Derived Van', 'Window Van', 'Camper'}
body_gt = lambda b: None if b is None or (isinstance(b, float)) or b in ('Manual', 'Limousine') else ('Van' if b in VAN else b)
# ---------- text side (computed once, cached) ----------
def text_embeds(df):
    cache = f'{OUT}/clip_text.npz'
    makers = sorted(df.Maker.astype(str).unique()); pairs = sorted({(m, g) for m, g in zip(df.Maker.astype(str), df.Genmodel.astype(str))})
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z['makers']) == makers and len(z['pairs']) == len(pairs): return makers, [tuple(p) for p in z['pairs']], z['em'], z['ep'], z['eb']
    tok = Tokenizer.from_file(f'{M}/tokenizer.json'); tok.enable_padding(length=77, pad_id=49407); tok.enable_truncation(77)
    s = ort.InferenceSession(f'{M}/text_model.onnx', providers=['CPUExecutionProvider'])
    def enc(texts):
        out = []
        for i in range(0, len(texts), 64):
            e = tok.encode_batch(texts[i:i+64]); ids = np.array([x.ids for x in e], np.int64); am = np.array([x.attention_mask for x in e], np.int64)
            feed = {'input_ids': ids}; 
            if 'attention_mask' in [x.name for x in s.get_inputs()]: feed['attention_mask'] = am
            out.append(s.run(['text_embeds'], feed)[0])
        v = np.concatenate(out); return v / np.linalg.norm(v, axis=1, keepdims=True)
    def ens(names, tmpls):
        v = np.mean([enc([t.format(n) for n in names]) for t in tmpls], axis=0); return v / np.linalg.norm(v, axis=1, keepdims=True)
    T = ['a photo of a {}.', 'a photo of a {}, a type of car.', 'a photo of my {}.', 'a close-up photo of a {} car.']
    em = ens(makers, T); ep = ens([f'{m} {g}' for m, g in pairs], T)
    eb = ens([BODY[k] for k in BODY], ['a photo of a {}.', 'a photo of a {} car.', 'a {} parked on the street.'])
    np.savez(cache, makers=np.array(makers), pairs=np.array(pairs, dtype=object), em=em, ep=ep, eb=eb); return makers, pairs, em, ep, eb
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32); STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
def prep(b):
    im = Image.open(io.BytesIO(b)).convert('RGB'); w, h = im.size; s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2
    x = (np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255 - MEAN) / STD; return x.transpose(2, 0, 1)
def main():
    df = pd.read_parquet(f'{D}/dvm.parquet'); df['Maker'] = df.Maker.astype(str); df['Bodytype'] = df.Bodytype.astype(object)
    makers, pairs, em, ep, eb = text_embeds(df); bk = list(BODY); pm = np.array([makers.index(m) for m, _ in pairs])
    meta = {r['Car Picture']: r for r in df[['Car Picture', 'Maker', 'Genmodel', 'Bodytype', 'Predicted_viewpoint', 'Reg_year', 'Color']].to_dict('records')}
    path = f'{OUT}/{a.tag}.csv'; done = set()
    if os.path.exists(path): done = {r['file'] for r in csv.DictReader(open(path))}
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1
    vs = ort.InferenceSession(f'{M}/vision_model{"_quantized" if a.model == "quantized" else ""}.onnx', so, providers=['CPUExecutionProvider'])
    new = not os.path.exists(path); fh = open(path, 'a', newline=''); w = csv.writer(fh)
    if new: w.writerow(['file', 'maker', 'genmodel', 'body', 'view', 'year', 'p_maker', 'p_maker_prob', 'top5_makers', 'p_pair', 'p_pair_prob', 'p_body', 'p_body_prob', 'ms', 'p_maker_agg', 'p_maker_agg_prob'])
    n = len(done); t0 = time.time(); k0 = n
    for z in sorted(glob.glob(f'{D}/dvm/images-*.zip')):
        try: zf = zipfile.ZipFile(z)
        except Exception as e: print('bad zip', z, e, flush=True); continue
        names = [x for x in zf.namelist() if x.endswith('.jpg') and x in meta and x not in done]
        for i in range(0, len(names), 16):
            if a.limit and n >= a.limit: break
            batch = names[i:i+16]; xs, ok = [], []
            for nm in batch:
                try: xs.append(prep(zf.read(nm))); ok.append(nm)
                except Exception as e: w.writerow([nm, meta[nm]['Maker'], meta[nm]['Genmodel'], body_gt(meta[nm]['Bodytype']), meta[nm]['Predicted_viewpoint'], meta[nm]['Reg_year'], 'ERR', 0, '', '', 0, '', 0, 0])
            if not xs: continue
            ts = time.time(); f = vs.run(['image_embeds'], {'pixel_values': np.stack(xs)})[0]; ms = (time.time() - ts) * 1000 / len(xs)
            f = f / np.linalg.norm(f, axis=1, keepdims=True)
            def sm(e): l = 100 * f @ e.T; l = l - l.max(1, keepdims=True); p = np.exp(l); return p / p.sum(1, keepdims=True)
            P_m, P_p, P_b = sm(em), sm(ep), sm(eb)
            P_a = np.zeros((len(ok), len(makers)), np.float32)
            for jj in range(len(ok)): np.add.at(P_a[jj], pm, P_p[jj])
            for j, nm in enumerate(ok):
                r = meta[nm]; t5 = np.argsort(-P_m[j])[:5]; pp = int(P_p[j].argmax()); pb = int(P_b[j].argmax())
                w.writerow([nm, r['Maker'], r['Genmodel'], body_gt(r['Bodytype']), r['Predicted_viewpoint'], r['Reg_year'], makers[t5[0]], round(float(P_m[j, t5[0]]), 4), '|'.join(makers[t] for t in t5),
                            f'{pairs[pp][0]}|{pairs[pp][1]}', round(float(P_p[j, pp]), 4), bk[pb], round(float(P_b[j, pb]), 4), round(ms, 1), makers[int(P_a[j].argmax())], round(float(P_a[j].max()), 4)])
            n += len(ok)
            if (n // 16) % 50 == 0:
                fh.flush(); el = time.time() - t0; print(f'{n} done  {(n-k0)/el:.1f} img/s  {os.path.basename(z)}', flush=True)
        fh.flush()
        if a.limit and n >= a.limit: break
    fh.close(); print('FINISHED', n, flush=True)
main()
