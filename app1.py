import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import io
from PIL import Image
import time
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import os

# For PyTorch model
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm

# For downloading models from Google Drive
import gdown

# Set page configuration
st.set_page_config(
    page_title="Soccer Fouls Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS for styling (same as before)
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .prediction-clean {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 5px solid #4CAF50;
    }
    .prediction-red {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 5px solid #F44336;
    }
    .prediction-yellow {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 5px solid #FFC107;
    }
    .card-yellow {
        background-color: rgba(255, 235, 59, 0.2);
        border-left: 5px solid #FFEB3B;
        font-weight: bold;
        color: #c7a600;
        padding: 0.7rem 1rem;
        border-radius: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .card-red {
        background-color: rgba(244, 67, 54, 0.13);
        border-left: 5px solid #F44336;
        font-weight: bold;
        color: #b80000;
        padding: 0.7rem 1rem;
        border-radius: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .card-none {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 5px solid #4CAF50;
        font-weight: bold;
        color: #357a38;
        padding: 0.7rem 1rem;
        border-radius: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .description {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .model-selector {
        padding: 1rem;
        background-color: rgba(25, 118, 210, 0.05);
        border-radius: 0.5rem;
        border: 1px solid rgba(25, 118, 210, 0.2);
        margin-bottom: 1.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title and description
st.markdown(
    '<h1 class="main-header">⚽ Soccer Fouls Analyzer with CNN</h1>', unsafe_allow_html=True
)

# Layout columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="description">', unsafe_allow_html=True)
    st.markdown(
        """
    Aplikasi ini menganalisa tackle pada sepak bola dan menentukan apakah termasuk handball sengaja, handball tidak sengaja, tackle bersih, tackle keras, atau tackle ringan.
    
    Upload gambar tackle, dan AI akan memprediksi hasil serta kemungkinan kartu.
    
    Model telah dilatih pada ratusan gambar fouls tackles dan handballs untuk mengenali perbedaan berdasarkan posisi pemain, titik kontak, dan faktor lainnya.
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<h3 class="sub-header">Cara Penggunaan</h3>', unsafe_allow_html=True)
    st.markdown(
        """
    1. Pilih model AI yang ingin digunakan
    2. Upload gambar fouls(tackle/handball) sepak bola
    3. Tunggu proses prediksi
    4. Lihat hasil prediksi, skor keyakinan, dan kemungkinan kartu
    5. Download hasil prediksi dalam format JPG atau PDF
    """
    )

# Sidebar with model info
with st.sidebar:
    st.header("Tentang Model")
    st.info(
        """
    Model ini menggunakan deep learning untuk klasifikasi tackle sepak bola.
    Kelas: Handball Sengaja, Handball Tidak Sengaja, Tackle Bersih, Tackle Keras, Tackle Ringan.
    """
    )
    st.markdown("### Detail Model")
    st.markdown("- **Available Models**: ResNet101V2 (TensorFlow), MobileNetV4 (PyTorch)")
    st.markdown("- **Classes**: Handball Sengaja, Handball Tidak Sengaja, Tackle Bersih, Tackle Keras, Tackle Ringan")
    st.markdown("- **Image Size**: 224x224 pixels")
    st.markdown("- **Possible Cards**: Merah, Kuning, Tidak Kartu")

# Classes/labels used by both models
LABELS = [
    "Handball Sengaja",
    "Handball Tidak Sengaja",
    "Tackle Bersih",
    "Tackle Keras",
    "Tackle Ringan"
]

# Google Drive file IDs for models
# GANTI DENGAN FILE ID ANDA!
RESNET_MODEL_ID = "YOUR_RESNET_FILE_ID_HERE"  # Ganti dengan ID Google Drive untuk ResNet
MOBILENET_MODEL_ID = "YOUR_MOBILENET_FILE_ID_HERE"  # Ganti dengan ID Google Drive untuk MobileNet

def download_model_from_gdrive(file_id, output_path):
    """Download model dari Google Drive jika belum ada"""
    if not os.path.exists(output_path):
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            st.info(f"📥 Downloading model {output_path}... (hanya dilakukan sekali)")
            gdown.download(url, output_path, quiet=False)
            st.success(f"✅ Model {output_path} berhasil didownload!")
            return True
        except Exception as e:
            st.error(f"❌ Gagal download model: {e}")
            return False
    return True

# Cache TensorFlow model loading
@st.cache_resource
def load_keras_model():
    try:
        model_path = "fould-classification-resnet101v2.h5"
        
        # Download model jika belum ada
        if not download_model_from_gdrive(RESNET_MODEL_ID, model_path):
            return None
        
        custom_objects = {}
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load TensorFlow model: {e}")
        return None

# Cache PyTorch model loading
@st.cache_resource
def load_pytorch_model():
    try:
        model_path = "best_model.pth"
        
        # Download model jika belum ada
        if not download_model_from_gdrive(MOBILENET_MODEL_ID, model_path):
            return None, None
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model('mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k', 
                                 pretrained=False, num_classes=len(LABELS))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        return model, device
    except Exception as e:
        st.error(f"Failed to load PyTorch model: {e}")
        return None, None

def preprocess_image_tensorflow(img_data):
    img = Image.open(io.BytesIO(img_data))
    img = img.resize((224, 224))
    img_array = np.array(img)
    if len(img_array.shape) < 3:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 1:
        img_array = np.concatenate([img_array] * 3, axis=2)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array, img

def preprocess_image_pytorch(img_data, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    return input_tensor, img

def predict_tackle(image_data, model_type="tensorflow"):
    try:
        if model_type == "tensorflow":
            model = load_keras_model()
            if model is None:
                return None, None
                
            img_array, _ = preprocess_image_tensorflow(image_data)
            result = model.predict(img_array)
            predicted_idx = np.argmax(result)
            predicted_class = LABELS[predicted_idx]
            confidence = float(np.max(result))
            
        elif model_type == "pytorch":
            model, device = load_pytorch_model()
            if model is None:
                return None, None
                
            input_tensor, _ = preprocess_image_pytorch(image_data, device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                
            predicted_idx = np.argmax(probs)
            predicted_class = LABELS[predicted_idx]
            confidence = float(probs[predicted_idx])
            
        else:
            st.error("Invalid model type")
            return None, None
            
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

def map_card(predicted_class, confidence):
    if predicted_class == "Handball Sengaja":
        return "Kartu Merah", "card-red", "🟥"
    elif predicted_class == "Handball Tidak Sengaja":
        return "Kartu Kuning", "card-yellow", "🟨"
    elif predicted_class == "Tackle Bersih":
        return "Tidak Kartu", "card-none", "🟢"
    elif predicted_class == "Tackle Keras":
        return "Kartu Merah", "card-red", "🟥"
    elif predicted_class == "Tackle Ringan":
        return "Kartu Kuning", "card-yellow", "🟨"
    else:
        return "Tidak Kartu", "card-none", "🟢"

def get_prediction_style(predicted_class):
    if predicted_class in ["Tackle Bersih"]:
        return "prediction-clean"
    elif predicted_class in ["Tackle Keras", "Handball Sengaja"]:
        return "prediction-red"
    elif predicted_class in ["Tackle Ringan", "Handball Tidak Sengaja"]:
        return "prediction-yellow"
    else:
        return ""

def create_analysis_image(img, predicted_class, confidence, card_label, card_icon, model_name):
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    ax_img = plt.subplot(gs[0])
    ax_img.imshow(img)
    ax_img.axis('off')
    
    ax_info = plt.subplot(gs[1])
    ax_info.axis('off')
    
    plt.suptitle("Soccer Fouls Analysis", fontsize=24, y=0.95, color="#1E88E5")
    
    info_text = f"""
    Model: {model_name}
    Prediksi: {predicted_class}
    Confidence Score: {confidence*100:.2f}%
    {card_icon} Kartu: {card_label}
    """
    ax_info.text(0.5, 0.5, info_text, 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    buf = BytesIO()
    plt.savefig(buf, format='jpg', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close()
    
    return buf

def generate_pdf_report(img_data, predicted_class, confidence, card_label, card_icon, model_name):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,
        textColor=colors.HexColor('#1E88E5')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=12,
        textColor=colors.HexColor('#424242')
    )
    
    elements.append(Paragraph("Soccer Fouls Analysis Report", title_style))
    elements.append(Spacer(1, 20))
    
    img = Image.open(BytesIO(img_data))
    orig_w, orig_h = img.size

    max_width = getattr(doc, 'width', doc.pagesize[0] - doc.leftMargin - doc.rightMargin)
    max_height = getattr(doc, 'height', doc.pagesize[1] - doc.topMargin - doc.bottomMargin)

    reserved_vertical = 200
    avail_height = max_height - reserved_vertical if (max_height - reserved_vertical) > 0 else max_height

    desired_w = min(400, int(max_width))
    desired_h = int((desired_w / orig_w) * orig_h)

    if desired_h > avail_height:
        desired_h = int(avail_height)
        desired_w = int((desired_h / orig_h) * orig_w)

    if desired_w > max_width:
        desired_w = int(max_width)
    if desired_h > max_height:
        desired_h = int(max_height)

    img = img.resize((max(1, desired_w), max(1, desired_h)))
    img_buffer = BytesIO()
    img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)

    elements.append(RLImage(img_buffer, width=desired_w, height=desired_h))
    elements.append(Spacer(1, 30))
    
    elements.append(Paragraph("Analysis Results", heading_style))
    elements.append(Spacer(1, 10))
    
    model_style = ParagraphStyle(
        'Model',
        parent=styles['BodyText'],
        fontSize=14,
        spaceAfter=8
    )
    elements.append(Paragraph(f"<b>Model Used:</b> {model_name}", model_style))
    
    pred_style = ParagraphStyle(
        'Prediction',
        parent=styles['BodyText'],
        fontSize=14,
        spaceAfter=8
    )
    
    elements.append(Paragraph(f"<b>Prediction:</b> {predicted_class}", pred_style))
    elements.append(Paragraph(f"<b>Confidence Score:</b> {confidence*100:.2f}%", pred_style))
    elements.append(Paragraph(f"<b>Card Decision:</b> {card_icon} {card_label}", pred_style))
    
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Italic']
    ))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

def get_binary_file_downloader_html(bin_data, file_label='File', file_name='file'):
    b64 = base64.b64encode(bin_data.read()).decode()
    custom_css = (
        '<style>'
        '.download-link {'
        '    display: inline-block;'
        '    padding: 0.8rem 1.5rem;'
        '    background-color: #1E88E5;'
        '    color: white !important;'
        '    text-decoration: none;'
        '    border-radius: 0.5rem;'
        '    font-weight: bold;'
        '    margin: 0.5rem 0;'
        '    text-align: center;'
        '}'
        '.download-link:hover {'
        '    background-color: #1565C0;'
        '}'
        '</style>'
    )
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}" class="download-link">{file_label}</a>'
    return custom_css + href

# Model selector
model_type = st.selectbox(
    "Pilih model AI untuk analisis:",
    [
        "ResNet101V2 (TensorFlow/Keras)", 
        "MobileNetV4 (PyTorch)"
    ]
)

st.markdown('<h3 class="sub-header">Upload Gambar Fouls Sepak Bola</h3>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

# Process model selection
if model_type == "ResNet101V2 (TensorFlow/Keras)":
    model_backend = "tensorflow"
    model_name = "ResNet101V2"
    model = load_keras_model()
else:
    model_backend = "pytorch"
    model_name = "MobileNetV4"
    model, _ = load_pytorch_model()

if uploaded_file is not None:
    image_data = uploaded_file.getvalue()
    image_display = Image.open(BytesIO(image_data))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image_display, caption="Gambar yang diupload", use_container_width=True)

    with col2:
        if model:
            with st.spinner("Menganalisa fouls..."):
                time.sleep(1)
                predicted_class, confidence = predict_tackle(image_data, model_backend)
                
                if predicted_class:
                    class_style = get_prediction_style(predicted_class)
                    card_label, card_style, card_icon = map_card(predicted_class, confidence)

                    st.markdown(f"### Menggunakan model: {model_name}")
                    st.markdown(f"### Prediksi: {predicted_class}")
                    st.markdown(f"**Confidence Score: {confidence*100:.2f}%**")
                    st.progress(confidence)

                    st.markdown(
                        f'<div class="{card_style}">{card_icon} Kemungkinan Kartu: <b>{card_label}</b></div>',
                        unsafe_allow_html=True
                    )

                    if predicted_class == "Tackle Bersih":
                        st.success("Ini adalah tackle bersih sesuai aturan permainan.")
                    elif predicted_class == "Tackle Keras":
                        st.error("Tackle keras terdeteksi, berpotensi mendapat kartu merah.")
                    elif predicted_class == "Tackle Ringan":
                        st.warning("Tackle ringan, kemungkinan kartu kuning.")
                    elif predicted_class == "Handball Sengaja":
                        st.error("Handball disengaja, berpotensi kartu merah.")
                    elif predicted_class == "Handball Tidak Sengaja":
                        st.warning("Handball tidak sengaja, kemungkinan kartu kuning.")

                    st.markdown("### Download Hasil Analisis")
                    
                    img_buf = create_analysis_image(
                        np.array(image_display),
                        predicted_class,
                        confidence,
                        card_label,
                        card_icon,
                        model_name
                    )
                    
                    pdf_buf = generate_pdf_report(
                        image_data,
                        predicted_class,
                        confidence,
                        card_label,
                        card_icon,
                        model_name
                    )
                    
                    download_format = st.selectbox(
                        "Pilih format unduhan:",
                        ("JPG", "PDF")
                    )

                    if download_format == "JPG":
                        st.markdown(
                            get_binary_file_downloader_html(
                                img_buf,
                                "📸 Download JPG",
                                f"soccer_tackle_analysis_{model_name}.jpg"
                            ),
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            get_binary_file_downloader_html(
                                pdf_buf,
                                "📄 Download PDF",
                                f"soccer_tackle_analysis_{model_name}.pdf"
                            ),
                            unsafe_allow_html=True
                        )
        else:
            st.error("Model tidak dimuat. Tidak dapat memprediksi.")

st.markdown("---")
st.markdown("© 2025 Soccer Fouls Analyzer | Created with Streamlit | Sendy Prismana Nurferian")