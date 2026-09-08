# RetinaAI — Explainable Diabetic Retinopathy Screening

RetinaAI is an educational/research prototype for diabetic retinopathy (DR) screening from retinal fundus images.

The system combines image-quality assessment, 5-class DR severity classification, confidence estimation, and Grad-CAM explainability in a lightweight web interface.

> ⚠️ **Disclaimer:** RetinaAI is an educational/research prototype and is not a medical diagnostic system. Results should not be used as a substitute for examination by a qualified ophthalmic professional.

## 🚀 Key Features

- 🖼️ Retinal fundus image upload
- 🔍 Image quality assessment
- 🧠 EfficientNet-B0 based DR classification
- 📊 5-stage DR severity prediction
- 📈 Class probability visualization
- 🔥 Grad-CAM explainability heatmap
- 🩺 Screening recommendation
- 📄 Downloadable screening report
- 👨‍⚕️ Tele-review workflow
- 📝 Reviewer notes and case status
- 🆔 Automatic case ID generation

## 🧠 DR Classification

The model predicts five stages:

| Class | Stage |
|---|---|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

## 🔬 System Workflow

```text
Retinal Image
      ↓
Image Quality Assessment
      ↓
EfficientNet-B0
      ↓
5-Class DR Classification
      ↓
Prediction + Confidence
      ↓
Grad-CAM Explainability
      ↓
Screening Result
      ↓
Report / Tele-Review
https://retinaai-diabetic-retinopathy-9tclga2a63thje7nbxrbut.streamlit.app/
