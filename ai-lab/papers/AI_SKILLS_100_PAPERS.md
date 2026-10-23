# 100 AI-improvement skills from 100 car-damage papers

Status: `have` 46 · `v2.1` 8 (this round) · `next` 27 (CPU, planned) · `gpu` 19 (needs retraining, v3). Paper numbers refer to `papers100_db.json` / the xlsx.

## A. Data & evaluation

1. `[v2.1]` Evaluate on the official CarDD test split (374 imgs, COCO box AP) so numbers are comparable with papers — papers #9, #1, #3
2. `[v2.1]` Report AP50 AND AP50:95 AND per-class AP; never a single accuracy number — papers #9, #1
3. `[have]` Report image-level recall + false-alarm rate on intact cars next to box AP (papers rarely do) — papers #9, #11
4. `[have]` Keep an out-of-domain test (VehiDE/AutoDamageIQ) besides in-domain CarDD — papers #76, #8
5. `[have]` Include intact close-ups as negatives (dealer photos are too easy) — papers #11, #9
6. `[have]` Use a damaged-vs-intact filter when collecting data (CarDD VGG16 filter: P 91.6 / R 97.6) — papers #9
7. `[next]` Remove near-duplicates across splits (Duplicate Cleaner) — papers #9
8. `[have]` Bootstrap 95% CIs for every headline metric — papers #8
9. `[v2.1]` Small/medium/large AP: small objects are where car-damage models fail (CarDD APs 37 vs APl 66) — papers #9
10. `[next]` Benchmark VLMs with structured (type, part, severity) tuples, not free text — papers #5, #10
11. `[next]` Separate hallucination rate for VLM answers (text-only 92% → grounded 31%) — papers #2
12. `[have]` Track every version on the same frozen sets (trend page) — papers #8
13. `[have]` Hard set with lamp/tyre and night/blur photos — papers #3, #11
14. `[next]` Collect consented Çekicin photos (KVKK) into a private dataset for retraining — papers #28, #33
15. `[next]` Label severity + part, not only type (insurer-style data) — papers #25, #39, #73

## B. Detector architecture & training (v3)

16. `[have]` YOLO11m is the best speed/accuracy trade-off on CarDD (63.3 AP50:95 vs RT-DETR-L) — papers #1, #4
17. `[gpu]` Licence-clean retrain on RF-DETR-M / D-FINE (Apache-2.0) — papers #12
18. `[gpu]` Multi-scale training (CarDD: +2.0 AP) — papers #9, #3
19. `[gpu]` Focal loss for hard classes crack/scratch (CarDD: +4.1 AP) — papers #9
20. `[gpu]` Deformable convolutions for irregular damage shapes (DCN+: best in CarDD paper) — papers #9, #35
21. `[gpu]` Attention modules (CBAM/SE/coordinate) in the YOLO neck — papers #4, #49, #23
22. `[gpu]` Heterogeneous / AKConv kernels for thin scratches — papers #49, #23
23. `[gpu]` NMS-free heads (YOLOv10) for lower latency — papers #23
24. `[gpu]` Dilated-conv FPN for multi-scale damage (DamageNet) — papers #35
25. `[gpu]` Diffusion-based detectors (DiffusionDet, C-DiffDet+) with scene context — papers #32, #90
26. `[gpu]` Region-aware mixture-of-experts for fine-grained classes — papers #16
27. `[gpu]` Instance segmentation (mask) for area-based severity — papers #18, #35, #68
28. `[gpu]` IoU-loss variants (CIoU/WIoU) for small boxes — papers #4, #49
29. `[gpu]` Synthetic / GAN-augmented damage images for rare classes — papers #47, #52
30. `[gpu]` Hyper-parameter search (e.g. Harris-Hawks-tuned YOLOv8-seg) — papers #53
31. `[gpu]` Knowledge distillation m → n for phones — papers #23, #49
32. `[have]` Train with intact close-ups as background images (hard negatives) — papers #11
33. `[gpu]` Copy-paste augmentation of damage onto intact panels — papers #47
34. `[gpu]` Self-/semi-supervised pre-training on unlabeled claim photos — papers #67
35. `[gpu]` Quantisation-aware training to keep int8 accuracy — papers #23

## C. Inference-time, no retraining

36. `[have]` Low detector threshold (0.05) + learned decision layer instead of a fixed 0.25 — papers #11
37. `[have]` Horizontal-flip test-time augmentation — papers #3
38. `[v2.1]` Weighted Boxes Fusion of TTA views instead of NMS (better boxes) — papers #1, #62
39. `[v2.1]` Evaluate at the COCO protocol (conf 0.001, top 300) AND at the shipped threshold — papers #9
40. `[have]` Class-aware NMS 0.45 (browser = Python parity) — papers #1
41. `[have]` Letterbox exactly like training (resize filter matters: measured) — papers #8
42. `[next]` Tiled / zoomed inference for small damage in wide shots (SAHI-style) — papers #9, #49
43. `[have]` Ask for a close-up when the car is small in the frame (wide crash scenes fail) — papers #7, #9
44. `[have]` Photo-quality gate: blur (Laplacian) + darkness → retake — papers #3, #59
45. `[next]` Glare/reflection check before accepting scratch/dent — papers #11, #2
46. `[have]` Ensemble the damage model with the parts model (features) — papers #62, #21
47. `[next]` Visual-embedding (CLIP) score as an extra decision feature — papers #62, #31
48. `[have]` Damaged-vs-intact global classifier as a pre-filter feature — papers #9, #21
49. `[have]` int8 quantised ONNX + WASM for on-device privacy — papers #23, #64
50. `[have]` Cache models in the browser; never upload the photo — papers #64
51. `[next]` Multi-scale TTA (640 + 800 px) for small damage — papers #9, #49
52. `[next]` Box-level ensemble of two damage detectors fused with WBF — papers #1, #62

## D. Calibration & decision

53. `[have]` Calibrated logistic stacker on 34 features (portable to JS/JSON) — papers #8
54. `[have]` Per-type heads with thresholds chosen inside CV folds only — papers #8
55. `[have]` False-alarm budget as an explicit constraint — papers #11
56. `[have]` Safety floor: lamp/tyre ≥ 0.60 always shown (tow-relevant) — papers #3
57. `[have]` Grouped CV so crops of one car never leak across folds — papers #8
58. `[have]` Confidence levels (high/medium/low) instead of raw scores in the UI — papers #5
59. `[have]` Abstain ("retake") instead of guessing on bad photos — papers #2
60. `[next]` Temperature/Platt check of the reported confidences (reliability diagram) — papers #8
61. `[next]` Per-class operating points tuned for business cost (missed tow ≫ false alarm) — papers #25
62. `[next]` Monitor drift: weekly score histograms on real traffic — papers #28
63. `[have]` Report AUROC damaged-vs-intact for the gate — papers #9, #21

## E. Parts & severity

64. `[have]` Damage + part localisation together (which part is damaged) — papers #17, #43, #98
65. `[v2.1]` Severity index from damage area / part area (DSI) — papers #3, #25
66. `[have]` Severity → action mapping: drivable / workshop / tow — papers #25, #39
67. `[next]` Three-quarter-view standard photos for consistent assessment — papers #43
68. `[next]` Repair-cost estimation from (type, part, severity) — papers #13, #15, #26
69. `[next]` Ontology / rule reasoning for cost (parts catalogue) — papers #82
70. `[have]` Glass damage pipeline separate from body damage — papers #59
71. `[next]` Tyre: flat vs. surface damage distinction — papers #84
72. `[have]` Count instances per type (2 areas of crack) in the result — papers #17
73. `[have]` Show boxes on the photo for transparency (explainability) — papers #54
74. `[next]` Damage-area / part-area ratio for repair-vs-replace hints — papers #25, #3

## F. VLM / LLM

75. `[have]` Ground the VLM on detector output (hypotheses to verify) → fewer hallucinations — papers #2, #50
76. `[have]` VLM only as a second opinion, never as the detector (weak localisation) — papers #5, #10
77. `[next]` Structured JSON answers from the VLM (type, part, severity, confidence) — papers #5
78. `[next]` Text-guided phrase grounding for user questions ("is the bumper cracked?") — papers #31
79. `[next]` Spatially-aware VLM prompting (regions, coordinates) — papers #50
80. `[next]` LLM-generated claim report from detections — papers #93, #98
81. `[next]` Fraud signals (re-used / edited photos) before claims — papers #46, #93
82. `[have]` Language-specific prompts (TR/EN) and answer checks — papers #5
83. `[have]` Voice questions answered by the assistant (Telegram) — papers #98
84. `[have]` Keep the human operator in the loop for the final decision — papers #10, #5

## G. Multi-photo / 3D

85. `[v2.1]` Multi-photo aggregation per incident (close-up + distant) — papers #7
86. `[next]` Close-up/distant pair alignment (patch-to-patch) — papers #67, #7
87. `[next]` Track damage across video frames (MOT) for walk-around videos — papers #87
88. `[gpu]` 3D lifting of damage (Gaussian splatting) for big repairs — papers #7
89. `[next]` Guided capture: ask for missing views (front/side/rear) — papers #43, #7
90. `[next]` De-duplicate the same damage seen in two photos of one incident — papers #7, #67

## H. Product, deployment, safety

91. `[have]` Real-time on phone (≤ 2–3 s) as in lightweight YOLO papers — papers #23, #49, #64
92. `[have]` Same decision layer on device, API, Telegram, WhatsApp (one JSON) — papers #28
93. `[have]` Public wiki + changelog for every version — papers #8
94. `[have]` Model cards with licence and training data (non-commercial CarDD risk) — papers #9, #12
95. `[have]` Clear disclaimer: pre-assessment, not an expert report — papers #10
96. `[have]` Explain WHY (advice text per type) not only WHAT — papers #54
97. `[have]` One-tap next action (call tow / workshop) — papers #28, #40
98. `[have]` A/B test new versions on the frozen benchmarks before shipping — papers #8
99. `[have]` Privacy: KVKK consent + no upload by default — papers #28
100. `[v2.1]` Publish a paper-style comparison page so partners can check claims — papers #8, #54

## Paper index (cited above)

#1: Car Damage Detection using YOLOv11 and RT-DETR (2025) https://doi.org/10.1109/isriti68345.2025.11393155
#2: Grounding Agentic VLMs with Dedicated Segmentation for Fine-Grained Vehicle Damage Assessment (2026) https://arxiv.org/abs/2608.02470
#3: Smart Car Damage Assessment Using Enhanced YOLO Algorithm and Image Processing Techniques (2025) https://doi.org/10.3390/info16030211
#4: Improved YOLOv11 algorithm for vehicle damage detection (2025) https://doi.org/10.1117/12.3071544
#5: CarDamageEval: Benchmark Evaluation of Car Damage Assessment Using Vision Language Models (2026) https://doi.org/10.1007/978-981-95-6786-7_17
#7: Car Damage Detection Based on Multi-View Fusion and Alignment: Dataset and Method (2025) https://doi.org/10.1109/tits.2025.3542174
#8: Vehicle Damage Detection Using Artificial Intelligence: A Systematic Literature Review (2025) https://doi.org/10.1002/widm.70027
#9: CarDD: A New Dataset for Vision-based Car Damage Detection (2022) https://arxiv.org/abs/2211.00945
#10: INS-MMBench: A Comprehensive Benchmark for Evaluating LVLMs' Performance in Insurance (2024) https://arxiv.org/abs/2406.09105
#11: Small Dents, Big Impact: A Dataset and Deep Learning Approach for Vehicle Dent Detection (2025) https://arxiv.org/abs/2508.15431v2
#12: RF-DETR: Neural Architecture Search for Real-Time Detection Transformers (2025) https://arxiv.org/abs/2511.09554
#13: Detects Damage Car Body using YOLO Deep Learning Algorithm (2023) https://doi.org/10.33395/sinkron.v8i2.12394
#15: CAR DAMAGE DETECTION USING DEEP LEARNING (2025) https://openalex.org/W7112198689
#16: CropResMoE-50: A Region-Aware Mixture-of-Experts Framework for Fine-Grained Vehicle Damage Detection and Semi-Automated Annotation (2026) https://doi.org/10.37965/jait.2026.0920
#17: Assessment of Intelligent Vehicle Damage and Cost Estimator for Insurance Companies (2023) https://doi.org/10.52783/tjjpt.v44.i4.1003
#18: Damaged Vehicle Parts Detection Platforms using Deep Learning Technique (2021) https://doi.org/10.37936/ecti-cit.2021153.223151
#21: Car Damage Detection Analysis using Deep Learning, Computer Vision Techniques (2025) https://doi.org/10.5120/icaidsc202404
#23: Lightweight Vehicle Damage Detection With NMS‐Free YOLOv10s and AKConv (2026) https://doi.org/10.1049/ipr2.70402
#25: AI-Based Vehicle Damage and Safety Analysis (2026) https://doi.org/10.5281/zenodo.21491792
#26: A System for Automated Vehicle Damage Localization and Severity Estimation Using Deep Learning (2025) https://doi.org/10.55041/ijsrem52093
#28: Cross-Architecture Performance Evaluation of Transfer Learning Models for Multi-Class Vehicle Damage Severity Classification (2026) https://doi.org/10.33395/sinkron.v10i2.15939
#31: GroundingCarDD: Text-Guided Multimodal Phrase Grounding for Car Damage Detection (2024) https://doi.org/10.1109/access.2024.3506563
#32: On the Application of DiffusionDet to Automatic Car Damage Detection and Classification via High-Performance Computing (2025) https://doi.org/10.3390/electronics14071362
#33: Implementation of Car Damage Detection in Insurance Claims (2024) https://doi.org/10.55041/ijsrem34585
#35: DamageNet: A Dilated Convolution Feature Pyramid Network Mask R‑CNN for Automated Car Damage Detection and Segmentation (2025) https://doi.org/10.14569/ijacsa.2025.0160521
#39: AI-Based Vehicle Damage Detection, Cost Estimation, and Insurance Claim Prediction System (2026) https://doi.org/10.65521/ijeecs.v15i1s.3100
#40: DEEP LEARNING BASED VEHICLE DAMAGE ASSESSMENT SYSTEM (2026) https://ijdim.com/journal/index.php/ijdim/article/download/1244/1182
#43: Automated vehicle damage classification using the three-quarter view car damage dataset and deep learning approaches (2024) https://doi.org/10.1016/j.heliyon.2024.e34016
#46: A novel data processing approach to detect fraudulent insurance claims for physical damage to cars (2022) https://doi.org/10.54187/jnrs.1129440
#47: From Synthetic Data to Real-World Efficiency: Automated Vehicle Damage Detection and Vehicle Parts Segmentation (2026) https://doi.org/10.1109/access.2026.3653645
#49: HL-YOLO: Improving Vehicle Damage Detection with Heterogeneous Convolutions and Large-Kernel Attention (2025) https://doi.org/10.3390/wevj16120640
#50: Enhanced spatial awareness of vision-language model for car damage detection (2025) https://doi.org/10.1117/1.jei.34.2.023037
#52: A Comparative Study of Generative Adversarial Networks for Generating Car Damaged Images (2023) https://doi.org/10.21203/rs.3.rs-2509412/v1
#53: Harris Hawks–tuned severity-aware YOLOv8 instance segmentation framework for vehicle damage assessment (2026) https://doi.org/10.1038/s41598-026-50925-0
#54: Deep Learning for Car Damage Detection: A Comprehensive Review (2025) https://doi.org/10.48175/ijarsct-30380
#59: A New Deep Learning Based Damage Classification Pipeline for Car Glass (2025) https://doi.org/10.28979/jarnas.1651526
#62: Vehicle Damage Detection with YOLO and Visual Embedding Fusion (2025) https://doi.org/10.6109/jkiice.2025.29.11.1481
#64: CarSenseAI: A Deep-Learning Pipeline for Real-Time Vehicle Damage Detection, Severity Assessment, and Repair-Cost Estimation (2026) https://doi.org/10.5281/zenodo.22765230
#67: Car Damage Detection and Patch-to-Patch Self-supervised Image Alignment (2024) https://doi.org/10.48550/arxiv.2403.06674
#68: Vehicle Intelligent Damage Assessment Based on Improved YOLOv8-seg (2025) https://doi.org/10.12677/mos.2025.144365
#73: Vehicle Damage Severity Estimation for Insurance Operations Using In-The-Wild Mobile Images (2023) https://doi.org/10.1109/access.2023.3299223
#76: Recognizing the Damaged Surface Parts of Cars in the Real Scene Using a Deep Learning Framework (2022) https://doi.org/10.1155/2022/5004129
#82: Enhancing car damage repair cost prediction: Integrating ontology reasoning with regression models (2024) https://doi.org/10.1016/j.iswa.2024.200411
#84: Vehicle Insurance Damage Detection (2023) https://doi.org/10.22214/ijraset.2023.49847
#87: Vehicle damage detection with transformer-based multi-object tracking (2026) https://doi.org/10.1016/j.engappai.2026.114725
#90: C-DiffDet+: Fusing Global Scene Context with Generative Denoising for High-Fidelity Car Damage Detection (2025) https://arxiv.org/abs/2509.00578v4
#93: YUNet_LLMClaimReport: An Enhanced Automobile Insurance Fraud Detection and Automated Claim Report Generation Using Large Language Models (2025) https://doi.org/10.1109/access.2025.3584927
#98: Integrating LLMs With Computer Vision: A Geometrically Invariant System for First Notice of Loss in Car Insurance (2026) https://doi.org/10.1111/exsy.70340