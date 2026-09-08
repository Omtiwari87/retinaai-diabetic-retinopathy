# RetinaAI — Explainable Diabetic Retinopathy Screening
## 🚀 Live Demo

👉 **[Open RetinaAI App](https://retinaai-diabetic-retinopathy-9tclga2a63thje7nbxrbut.streamlit.app/)**

Click the link above to open the RetinaAI diabetic retinopathy screening application.

The application is deployed on Streamlit Community Cloud.

### How to use

1. Open the live app using the link above.
2. Upload a clear retinal/fundus image.
3. Click **Analyze Image**.
4. View the predicted DR stage, confidence and Grad-CAM heatmap.

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


