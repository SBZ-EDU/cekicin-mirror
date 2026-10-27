---
license: agpl-3.0
library_name: onnx
pipeline_tag: object-detection
base_model: vineetsarpal/yolov11n-car-damage
tags: [car-damage, yolo11, onnx, onnxruntime-web, roadside-assistance, cekicin]
datasets: [sosa123454321/cekicin-ai-knowledge]
---

# Çekicin car-damage detector (YOLO11n, ONNX)

ONNX export of [`vineetsarpal/yolov11n-car-damage`](https://huggingface.co/vineetsarpal/yolov11n-car-damage), packaged
to run **in the browser** with `onnxruntime-web`. Used by the
**[Çekicin AI Lab Space](https://huggingface.co/spaces/sosa123454321/cekicin-ai-lab)**.

- **Architecture:** YOLO11-nano, 2.58 M parameters, 6.4 GFLOPs
- **Input:** `images` float32 `[1,3,640,640]`, RGB, ÷255, letterbox with grey (114) padding
- **Output:** `[1,18,8400]`: `cx, cy, w, h` + 14 class scores; apply a score threshold and NMS yourself
- **Reported accuracy (source model):** mAP@0.5 = 79.7 %, mAP@0.5:0.95 = 55.2 % on the Roboflow
  "Automobile Damage Detection" dataset (~6.9 k images)
- **Export:** Ultralytics 8.4, opset 12, onnxslim. Checked against the PyTorch weights on test photos (same top classes).

## Classes → car part

`labels.json` maps each class to a car part and a damage type, with Turkish/English labels. The repair/replace rules and
cost table live in the dataset [`sosa123454321/cekicin-ai-knowledge`](https://huggingface.co/datasets/sosa123454321/cekicin-ai-knowledge).

| id | label | part | TR |
|---|---|---|---|
| 0 | Front-windscreen-damage | windscreen_front | Ön cam hasarı |
| 1 | Headlight-damage | headlight | Far hasarı |
| 2 | Rear-windscreen-Damage | windscreen_rear | Arka cam hasarı |
| 3 | Runningboard-Damage | running_board | Marşpiyel hasarı |
| 4 | Sidemirror-Damage | side_mirror | Yan ayna hasarı |
| 5 | Taillight-Damage | taillight | Stop lambası hasarı |
| 6 | bonnet-dent | bonnet | Kaput göçüğü |
| 7 | boot-dent | boot | Bagaj kapağı göçüğü |
| 8 | doorouter-dent | door | Kapı göçüğü |
| 9 | fender-dent | fender | Çamurluk göçüğü |
| 10 | front-bumper-dent | bumper_front | Ön tampon hasarı |
| 11 | quaterpanel-dent | quarter_panel | Arka çamurluk göçüğü |
| 12 | rear-bumper-dent | bumper_rear | Arka tampon hasarı |
| 13 | roof-dent | roof | Tavan göçüğü |

## Licence note

The source repository is tagged Apache-2.0, but YOLO11 weights trained with Ultralytics fall under **AGPL-3.0** unless
you hold an Ultralytics Enterprise licence, so this export is published as AGPL-3.0. That is fine for this public, open-source
demo. To ship it inside the closed Çekicin mobile app, either buy an Ultralytics licence or switch to an
Apache-2.0 detector (for example an RT-DETR model trained with 🤗 Transformers).

## Limits

The detector only knows the 14 classes above. It does not see scratches, mechanical faults or hidden damage. Treat
its output as a **pre-assessment**, not an expert report.
