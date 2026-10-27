# Çekicin AI Lab

`index.html` is served at **https://cekicin.pages.dev/ai-lab/** and is the same file as the Hugging Face Space
**https://huggingface.co/spaces/sosa123454321/cekicin-ai-lab**.

| Folder | Published to |
|---|---|
| `index.html` (+ `SPACE_README.md` as the Space's `README.md`) | Space `sosa123454321/cekicin-ai-lab` (static) |
| `dataset/` | Dataset `sosa123454321/cekicin-ai-knowledge`; the page reads these CSVs at runtime |
| `model/` (card, labels, export script; the 10 MB `model.onnx` is only on the Hub) | Model `sosa123454321/cekicin-damage-yolo11n-onnx` |

Modules: damage detection (YOLO11n / onnxruntime-web) → repair/replace + cost (dataset rules), nearby services &
parts suppliers (OpenStreetMap), voice assistant (Whisper tiny → multilingual MiniLM RAG → speech synthesis),
impact calculator (UK DESNZ 2025 factors). Everything runs client-side.

To change prices, rules or FAQ answers, edit the CSV on the Hub dataset (or here, then upload). No redeploy needed.
