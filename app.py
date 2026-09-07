from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

import io

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="RetinaAI — Diabetic Retinopathy Screening",
    page_icon="👁️",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    color: #666;
    font-size: 17px;
    margin-bottom: 25px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="main-title">👁️ RetinaAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-assisted Diabetic Retinopathy Screening</div>',
    unsafe_allow_html=True
)

st.info(
    "⚠️ Educational/research prototype only. "
    "This system is not a medical diagnosis."
)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        5
    )

    model_path = "best_retinopathy_model.pth"

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device


model, device = load_model()


class_names = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


stage_info = {
    "No DR": "No diabetic retinopathy detected by the model.",
    "Mild DR": "Early-stage diabetic retinopathy pattern.",
    "Moderate DR": "Moderate-stage diabetic retinopathy pattern.",
    "Severe DR": "Advanced diabetic retinopathy pattern.",
    "Proliferative DR": "Advanced proliferative-stage pattern."
}


# =========================================================
# PREPROCESSING
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# =========================================================
# IMAGE QUALITY ASSESSMENT (IQA)
# =========================================================
def check_image_quality(image):
    image_array = np.array(image)

    gray = np.array(
        Image.fromarray(image_array).convert("L"),
        dtype=np.float32
    )

    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)

    blur_score = float(
        (np.var(dx) + np.var(dy)) / 2
    )

    contrast_score = float(gray.std())

    # Less strict threshold for cloud deployment
    quality_ok = (
        blur_score >= 5 and
        contrast_score >= 15
    )

    return quality_ok, blur_score, contrast_score
# =========================================================
# AUTOMATED PDF REPORT
# =========================================================

def create_pdf_report(
    predicted_name,
    confidence,
    probability_data,
    blur_score,
    contrast_score
):

    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "RetinaAI — Diabetic Retinopathy Screening Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            f"Report Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            f"<b>Predicted Stage:</b> {predicted_name}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Model Confidence:</b> {confidence:.2f}%",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "<b>Image Quality Assessment</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"Blur Score: {blur_score:.2f}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Contrast Score: {contrast_score:.2f}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "<b>Class Probabilities</b>",
            styles["Heading2"]
        )
    )

    table_data = [
        ["DR Stage", "Probability"]
    ]

    for _, row in probability_data.iterrows():

        table_data.append([
            row["Stage"],
            f"{row['Probability (%)']:.2f}%"
        ])

    table = Table(table_data)

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8)
        ])
    )

    story.append(table)

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This report is generated by an "
            "educational/research prototype. Model confidence should "
            "not be interpreted as medical certainty. This system "
            "is not a medical diagnosis.",
            styles["Normal"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer

# =========================================================
# UPLOAD
# =========================================================

st.subheader("📤 Upload Retinal Image")

uploaded_file = st.file_uploader(
    "Choose a fundus image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    # Image Quality Assessment
    quality_ok, blur_score, contrast_score = check_image_quality(image)

    if not quality_ok:
        st.error(
            "❌ Poor image quality. "
            "Please upload a clearer retinal image."
        )
        st.stop()
        # IMAGE QUALITY RESULT

    st.success("✅ Image Quality: PASS")

    quality_col1, quality_col2 = st.columns(2)

    with quality_col1:
        st.metric(
            "Blur Score",
            f"{blur_score:.2f}"
        )

    with quality_col2:
        st.metric(
            "Contrast Score",
            f"{contrast_score:.2f}"
        )

    # Basic image validation
    width, height = image.size

    if width < 100 or height < 100:
        st.error("❌ Image is too small. Please upload a clear retinal/fundus image.")
        st.stop()
        # Blank / black image validation
    image_array = np.array(image)

    mean_brightness = image_array.mean()

    if mean_brightness < 15:
        st.error(
            "❌ Invalid image. The uploaded image appears to be blank or too dark. "
            "Please upload a clear retinal/fundus image."
        )
        st.stop()
        # Fundus image quality check
    check_image = image.resize((128, 128))
    arr = np.array(check_image).astype(np.float32)

    # Check center brightness
    center = arr[32:96, 32:96].mean()

    # Check corner darkness
    corners = np.concatenate([
        arr[:24, :24].reshape(-1, 3),
        arr[:24, -24:].reshape(-1, 3),
        arr[-24:, :24].reshape(-1, 3),
        arr[-24:, -24:].reshape(-1, 3)
    ])

    corner_brightness = corners.mean()

    # Retinal images usually contain noticeable reddish tones
    red_score = (
        arr[:, :, 0] -
        (arr[:, :, 1] + arr[:, :, 2]) / 2
    ).mean()

    fundus_score = 0

    if center > 35:
        fundus_score += 1

    if corner_brightness < center:
        fundus_score += 1

    if red_score > 5:
        fundus_score += 1

    if fundus_score < 3:
        st.error(
            "❌ Invalid image. This does not appear to be a "
            "clear retinal/fundus image. Please upload a proper fundus photograph."
        )
        st.stop()

    st.caption(f"Image size: {width} × {height} pixels")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded retinal image",
            use_container_width=True
        )

    with col2:

        st.markdown("### Ready for analysis")

        st.write(
            "The uploaded image will be processed by "
            "the trained EfficientNet-B0 model."
        )

        analyze = st.button(
            "🔍 Analyze Image",
            use_container_width=True
        )


    if analyze:

        # =================================================
        # PREDICTION
        # =================================================

        input_tensor = transform(
            image
        ).unsqueeze(0).to(device)


        with torch.no_grad():

            output = model(input_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]


        predicted_class = torch.argmax(
            probabilities
        ).item()

        confidence = (
            probabilities[predicted_class].item()
            * 100
        )

        predicted_name = class_names[predicted_class]


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader("📊 Screening Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.success(
                f"### Predicted Stage\n"
                f"{predicted_name}"
            )

        with result_col2:

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )


        st.write(stage_info[predicted_name])

        st.caption(
            "Model confidence should not be interpreted "
            "as medical certainty."
        )
        # SCREENING RECOMMENDATION

        st.subheader("🩺 Screening Recommendation")

        if predicted_class == 0:
            st.success(
                "Low-risk screening result. "
                "Routine eye screening is recommended."
            )
        else:
            st.warning(
                "DR-related pattern detected. "
                "Professional ophthalmic review is recommended."
            )


        # =================================================
        # PROBABILITY
        # =================================================

        st.subheader("📈 Class Probabilities")

        probability_data = pd.DataFrame({
            "Stage": class_names,
            "Probability (%)": [
                round(p.item() * 100, 2)
                for p in probabilities
            ]
        })

        st.bar_chart(
            probability_data.set_index("Stage")
        )
        # =================================================
# DOWNLOAD PDF REPORT
# =================================================
# CREATE PDF REPORT

    report_pdf = create_pdf_report(
        predicted_name,
        confidence,
        probability_data,
        blur_score,
        contrast_score
    )

    st.download_button(
        label="📄 Download Screening Report",
        data=report_pdf,
        file_name="RetinaAI_Screening_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    # =================================================
    # TELE-REVIEW
    # =================================================

    st.subheader("👨‍⚕️ Tele-Review")

    review_status = st.selectbox(
        "Review Status",
        [
            "Pending Specialist Review",
            "Reviewed",
            "Needs Re-capture"
        ]
    )

    st.info(
        "This screening case can be shared with a qualified "
        "ophthalmic reviewer for further assessment."
    )
    st.markdown("### 📋 Case Summary")

    st.write(f"**Predicted Stage:** {predicted_name}")
    st.write(f"**Model Confidence:** {confidence:.2%}")
    st.write(f"**Image Quality:** PASS")
    st.write(f"**Review Status:** {review_status}")
    st.markdown("### 📝 Reviewer Notes")

    reviewer_notes = st.text_area(
        "Add notes for the ophthalmic reviewer",
        placeholder="Enter observations or follow-up notes..."
    )
    case_id = datetime.now().strftime("RETINA-%Y%m%d-%H%M%S")

    st.caption(f"Case ID: {case_id}")
    if review_status == "Needs Re-capture":
        st.error("⚠️ Please capture and upload a clearer retinal image.")
    elif review_status == "Reviewed":
        st.success("✅ Case marked as reviewed.")
    else:
        st.info("⏳ Case is pending specialist review.")


    # =================================================
    # GRAD-CAM
    # =================================================
    st.subheader("🔎 Explainable AI — Grad-CAM")    

    target_layers = [
        model.features[-1]
    ]

    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]

    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    display_image = image.resize(
        (224, 224)
    )

    rgb_image = (
        np.array(display_image)
        .astype(np.float32)
        / 255.0
    )

    cam_image = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    st.image(
        cam_image,
        caption="Grad-CAM: Areas influencing the prediction",
        use_container_width=True
    )

    st.caption(
        "Highlighted regions show areas that contributed "
        "more strongly to the model prediction."
    )
        





# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "RetinaAI • EfficientNet-B0 • Grad-CAM • "
    "Educational/Research Prototype"
)
