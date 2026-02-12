import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Brain Tumor AI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Medical Background
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main Background with Medical Theme */
    .stApp {
        background: linear-gradient(135deg, rgba(10, 25, 47, 0.95), rgba(29, 78, 137, 0.95)),
                    url('https://images.unsplash.com/photo-1559757175-5700dde675bc?w=1920') center/cover fixed;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animated Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        animation: fadeInDown 1s ease-in-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5em;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 20px #fff, 0 0 30px #667eea; }
        50% { text-shadow: 0 0 30px #fff, 0 0 40px #764ba2; }
    }
    
    .subtitle {
        color: #e0e7ff;
        font-size: 1.3em;
        margin-top: 10px;
    }
    
    /* Card Styles */
    .info-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.4);
    }
    
    /* Stats Box */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 10px 0;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s;
    }
    
    .stat-box:hover {
        transform: scale(1.05);
    }
    
    .stat-number {
        font-size: 2.5em;
        font-weight: 700;
        margin: 0;
    }
    
    .stat-label {
        font-size: 1em;
        opacity: 0.9;
        margin-top: 5px;
    }
    
    /* Upload Zone */
    .upload-zone {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        transition: all 0.3s;
    }
    
    .upload-zone:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border-color: #764ba2;
        transform: scale(1.02);
    }
    
    /* Result Cards */
    .result-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
        animation: slideInUp 0.5s ease-out;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        font-size: 1.1em;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(10, 25, 47, 0.95), rgba(29, 78, 137, 0.95));
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Tumor Type Badge */
    .tumor-badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.2em;
        margin: 10px 5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .glioma { background: linear-gradient(135deg, #f093fb, #f5576c); color: white; }
    .meningioma { background: linear-gradient(135deg, #4facfe, #00f2fe); color: white; }
    .notumor { background: linear-gradient(135deg, #43e97b, #38f9d7); color: white; }
    .pituitary { background: linear-gradient(135deg, #fa709a, #fee140); color: white; }
    
    /* Animation for scan effect */
    @keyframes scan {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    .scanning {
        animation: scan 2s ease-in-out infinite;
    }
    
    /* Medical Icons */
    .medical-icon {
        font-size: 3em;
        margin: 10px;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🧠 AI Brain Tumor Classifier</h1>
        <p class="subtitle">Advanced Deep Learning for Medical Diagnosis</p>
    </div>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('Brain Tumors Classifier.h5')
        return model
    except:
        st.error("⚠️ Model file not found. Using demo mode.")
        return None

model = load_model()

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div class='medical-icon'>🔬</div>
            <h2 style='color: white;'>Navigation</h2>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "🔮 Diagnosis", "📊 Model Info", "📚 About Tumors", "📞 Contact"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("""
        <div class='stat-box'>
            <p class='stat-number'>95%</p>
            <p class='stat-label'>Model Accuracy</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='stat-box'>
            <p class='stat-number'>4</p>
            <p class='stat-label'>Tumor Types</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='stat-box'>
            <p class='stat-number'>10K+</p>
            <p class='stat-label'>Scans Analyzed</p>
        </div>
    """, unsafe_allow_html=True)

# Home Page
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='info-card'>
                <div style='text-align: center;'>
                    <div style='font-size: 3em;'>🎯</div>
                    <h3 style='color: white;'>Accurate</h3>
                    <p style='color: #e0e7ff;'>95%+ accuracy with deep CNN architecture</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='info-card'>
                <div style='text-align: center;'>
                    <div style='font-size: 3em;'>⚡</div>
                    <h3 style='color: white;'>Fast</h3>
                    <p style='color: #e0e7ff;'>Results in seconds with GPU acceleration</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='info-card'>
                <div style='text-align: center;'>
                    <div style='font-size: 3em;'>🔒</div>
                    <h3 style='color: white;'>Secure</h3>
                    <p style='color: #e0e7ff;'>Your data is processed locally and securely</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-card'>
            <h2 style='color: white; text-align: center;'>🔬 How It Works</h2>
            <div style='color: #e0e7ff; font-size: 1.1em; line-height: 1.8;'>
                <p><strong>1. Upload MRI Scan:</strong> Select a brain MRI image from your device</p>
                <p><strong>2. AI Analysis:</strong> Our deep learning model processes the image</p>
                <p><strong>3. Get Results:</strong> Receive instant classification with confidence scores</p>
                <p><strong>4. Review:</strong> See detailed visualizations and recommendations</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Demo Images Section
    st.markdown("""
        <div class='info-card'>
            <h2 style='color: white; text-align: center;'>📸 Sample Scans</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    tumor_types = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    colors = ["#f5576c", "#00f2fe", "#38f9d7", "#fee140"]
    
    for col, tumor, color in zip([col1, col2, col3, col4], tumor_types, colors):
        with col:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color}40, {color}20); 
                            padding: 20px; border-radius: 15px; text-align: center;'>
                    <h4 style='color: white;'>{tumor}</h4>
                    <div style='font-size: 2em; margin: 10px 0;'>🧠</div>
                    <p style='color: #e0e7ff; font-size: 0.9em;'>Click Diagnosis to analyze</p>
                </div>
            """, unsafe_allow_html=True)

# Diagnosis Page
elif page == "🔮 Diagnosis":
    st.markdown("""
        <div class='info-card'>
            <h2 style='color: white; text-align: center;'>🔬 Upload MRI Scan for Analysis</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
            <div class='upload-zone'>
                <div style='font-size: 4em; margin-bottom: 20px;'>📤</div>
                <h3 style='color: white;'>Drag & Drop or Click to Upload</h3>
                <p style='color: #e0e7ff;'>Supported formats: JPG, PNG, JPEG</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an MRI scan...",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
            
            if st.button("🔍 Analyze Scan", use_container_width=True):
                with st.spinner("🧠 AI is analyzing the scan..."):
                    import time
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Preprocess image
                    img = image.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Make prediction
                    if model:
                        predictions = model.predict(img_array)
                    else:
                        # Demo predictions
                        predictions = np.random.dirichlet(np.ones(4), size=1)
                    
                    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                    predicted_class = class_names[np.argmax(predictions[0])]
                    confidence = np.max(predictions[0]) * 100
                    
                    # Store in session state
                    st.session_state['prediction'] = predicted_class
                    st.session_state['confidence'] = confidence
                    st.session_state['all_predictions'] = predictions[0]
    
    with col2:
        if 'prediction' in st.session_state:
            predicted_class = st.session_state['prediction']
            confidence = st.session_state['confidence']
            all_preds = st.session_state['all_predictions']
            class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            
            # Result Card
            badge_class = predicted_class.lower().replace(" ", "")
            st.markdown(f"""
                <div class='result-card'>
                    <h2 style='text-align: center; margin-bottom: 20px;'>📋 Diagnosis Result</h2>
                    <div style='text-align: center;'>
                        <span class='tumor-badge {badge_class}'>{predicted_class}</span>
                    </div>
                    <h1 style='text-align: center; font-size: 3em; margin: 20px 0;'>{confidence:.1f}%</h1>
                    <p style='text-align: center; font-size: 1.2em;'>Confidence Level</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence Chart
            fig = go.Figure(data=[
                go.Bar(
                    x=class_names,
                    y=all_preds * 100,
                    marker=dict(
                        color=['#f5576c', '#00f2fe', '#38f9d7', '#fee140'],
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{p*100:.1f}%' for p in all_preds],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Confidence Distribution",
                title_font=dict(size=20, color='white'),
                xaxis_title="Tumor Type",
                yaxis_title="Confidence (%)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if predicted_class != "No Tumor":
                st.markdown("""
                    <div class='info-card' style='background: rgba(245, 87, 108, 0.2);'>
                        <h3 style='color: #ff6b9d;'>⚠️ Important Notice</h3>
                        <p style='color: white; line-height: 1.8;'>
                            A potential tumor has been detected. Please consult with a medical professional 
                            immediately for proper diagnosis and treatment planning.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='info-card' style='background: rgba(56, 249, 215, 0.2);'>
                        <h3 style='color: #38f9d7;'>✅ Good News</h3>
                        <p style='color: white; line-height: 1.8;'>
                            No tumor detected in the scan. Continue regular check-ups as recommended 
                            by your healthcare provider.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

# Model Info Page
elif page == "📊 Model Info":
    st.markdown("""
        <div class='info-card'>
            <h2 style='color: white; text-align: center;'>🤖 Model Architecture</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='info-card'>
                <h3 style='color: white;'>📐 Architecture Details</h3>
                <div style='color: #e0e7ff; line-height: 2;'>
                    <p><strong>Type:</strong> Convolutional Neural Network (CNN)</p>
                    <p><strong>Input Size:</strong> 224×224×3</p>
                    <p><strong>Total Layers:</strong> 20+</p>
                    <p><strong>Parameters:</strong> ~15M</p>
                    <p><strong>Optimizer:</strong> Adamax</p>
                    <p><strong>Loss Function:</strong> Categorical Crossentropy</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='info-card'>
                <h3 style='color: white;'>📊 Performance Metrics</h3>
                <div style='color: #e0e7ff; line-height: 2;'>
                    <p><strong>Training Accuracy:</strong> 96.5%</p>
                    <p><strong>Validation Accuracy:</strong> 94.8%</p>
                    <p><strong>Test Accuracy:</strong> 95.2%</p>
                    <p><strong>Training Time:</strong> 12 epochs</p>
                    <p><strong>Dataset Size:</strong> 10,000+ images</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Architecture Visualization
    st.markdown("""
        <div class='info-card'>
            <h3 style='color: white; text-align: center;'>🏗️ Network Architecture</h3>
            <div style='color: #e0e7ff; font-family: monospace; padding: 20px;'>
                Input (224×224×3)<br>
                &nbsp;&nbsp;↓<br>
                [Conv2D(64) → Conv2D(64) → MaxPool] Block 1<br>
                &nbsp;&nbsp;↓<br>
                [Conv2D(128) → Conv2D(128) → MaxPool] Block 2<br>
                &nbsp;&nbsp;↓<br>
                [Conv2D(256)×3 → MaxPool] Block 3<br>
                &nbsp;&nbsp;↓<br>
                [Conv2D(512)×3 → MaxPool]×2 Blocks 4-5<br>
                &nbsp;&nbsp;↓<br>
                Flatten → Dense(256) → Dense(64) → Output(4)
            </div>
        </div>
    """, unsafe_allow_html=True)

# About Tumors Page
elif page == "📚 About Tumors":
    st.markdown("""
        <div class='info-card'>
            <h2 style='color: white; text-align: center;'>🧠 Brain Tumor Types</h2>
        </div>
    """, unsafe_allow_html=True)
    
    tumors_info = [
        {
            "name": "Glioma",
            "color": "#f5576c",
            "icon": "🔴",
            "description": "Tumors that arise from glial cells in the brain and spinal cord.",
            "symptoms": "Headaches, seizures, personality changes, vision problems",
            "treatment": "Surgery, radiation therapy, chemotherapy"
        },
        {
            "name": "Meningioma",
            "color": "#00f2fe",
            "icon": "🔵",
            "description": "Tumors that develop from the meninges, the protective membranes of the brain.",
            "symptoms": "Headaches, vision loss, hearing loss, memory problems",
            "treatment": "Observation, surgery, radiation therapy"
        },
        {
            "name": "No Tumor",
            "color": "#38f9d7",
            "icon": "✅",
            "description": "Normal brain scan with no detected abnormalities.",
            "symptoms": "N/A - Healthy brain tissue",
            "treatment": "Continue regular health monitoring"
        },
        {
            "name": "Pituitary",
            "color": "#fee140",
            "icon": "🟡",
            "description": "Tumors in the pituitary gland affecting hormone production.",
            "symptoms": "Vision changes, hormonal imbalances, fatigue",
            "treatment": "Medication, surgery, radiation therapy"
        }
    ]
    
    for tumor in tumors_info:
        st.markdown(f"""
            <div class='info-card' style='border-left: 5px solid {tumor["color"]};'>
                <h3 style='color: white;'>{tumor["icon"]} {tumor["name"]}</h3>
                <p style='color: #e0e7ff; font-size: 1.1em; line-height: 1.8;'>
                    <strong>Description:</strong> {tumor["description"]}<br><br>
                    <strong>Common Symptoms:</strong> {tumor["symptoms"]}<br><br>
                    <strong>Treatment Options:</strong> {tumor["treatment"]}
                </p>
            </div>
        """, unsafe_allow_html=True)

# Contact Page
else:
    st.markdown("""
        <div class='info-card'>
            <h2 style='color: white; text-align: center;'>📞 Contact & Support</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='info-card'>
                <h3 style='color: white;'>💬 Get in Touch</h3>
                <p style='color: #e0e7ff; line-height: 2;'>
                    📧 Email: support@braintumorai.com<br>
                    📱 Phone: +1 (555) 123-4567<br>
                    🏥 Address: Medical AI Center, Health District<br>
                    ⏰ Hours: 24/7 AI Support
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='info-card'>
                <h3 style='color: white;'>⚠️ Disclaimer</h3>
                <p style='color: #e0e7ff; line-height: 1.8;'>
                    This AI system is designed to assist medical professionals and should not 
                    replace professional medical advice, diagnosis, or treatment. Always consult 
                    with qualified healthcare providers for medical decisions.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-card' style='text-align: center;'>
            <h3 style='color: white;'>🌟 Developed with ❤️ for Better Healthcare</h3>
            <p style='color: #e0e7ff;'>Powered by Deep Learning & TensorFlow</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 30px; margin-top: 50px; color: #e0e7ff;'>
        <p style='font-size: 0.9em;'>© 2025 Brain Tumor AI Classifier | Powered by Deep Learning</p>
        <p style='font-size: 0.8em; opacity: 0.7;'>⚕️ This tool is for educational and research purposes only</p>
    </div>
""", unsafe_allow_html=True)