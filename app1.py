import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import time
import os
import cv2
from PIL import Image
import tempfile
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ==============================================
# INITIALIZATION & CONFIGURATION
# ==============================================

# Constants
HR_CSV = "HR_alert.csv"
STRESS_CSV = "stress_history.csv"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Set page configuration
st.set_page_config(
    page_title="Emotion & Stress Detector", 
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .st-emotion-card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .st-recommendation {
        background-color: #f8f9fa;
        border-left: 4px solid #4e73df;
        padding: 10px;
        margin: 10px 0;
    }
    .st-alert {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .st-alert-success {
        background-color: #d4edda;
        color: #155724;
    }
    .st-alert-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    .st-alert-danger {
        background-color: #f8d7da;
        color: #721c24;
    }
    .camera-feed {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Emotion dictionary with emojis and colors
emotions_emoji_dict = {
    "anger": {"emoji": "😠", "color": "#dc3545"},
    "disgust": {"emoji": "🤮", "color": "#28a745"},
    "fear": {"emoji": "😨", "color": "#6c757d"},
    "happy": {"emoji": "😊", "color": "#ffc107"},
    "joy": {"emoji": "😂", "color": "#fd7e14"},
    "sad": {"emoji": "😔", "color": "#17a2b8"},
    "neutral": {"emoji": "😐", "color": "#6c757d"},
    "angry": {"emoji": "😠", "color": "#dc3545"},
    "calm": {"emoji": "😌", "color": "#28a745"},
    "surprise": {"emoji": "😮", "color": "#6610f2"}
}

# Initialize data storage
def init_data_storage():
    if not os.path.exists(HR_CSV) or os.path.getsize(HR_CSV) == 0:
        pd.DataFrame(columns=["timestamp", "user_id", "alert_type", "message"]).to_csv(HR_CSV, index=False)
    
    if not os.path.exists(STRESS_CSV) or os.path.getsize(STRESS_CSV) == 0:
        pd.DataFrame(columns=["user_id", "timestamp", "text", "stress_level"]).to_csv(STRESS_CSV, index=False)
    
    if 'stress_history' not in st.session_state:
        st.session_state.stress_history = pd.read_csv(STRESS_CSV) if os.path.exists(STRESS_CSV) else pd.DataFrame(columns=["user_id", "timestamp", "text", "stress_level"])
    
    if 'hr_alerts' not in st.session_state:
        st.session_state.hr_alerts = pd.read_csv(HR_CSV) if os.path.exists(HR_CSV) else pd.DataFrame(columns=["timestamp", "user_id", "alert_type", "message"])

init_data_storage()

# ==============================================
# MODEL LOADING (YOUR CUSTOM MODELS + VOICE PRE-TRAINED)
# ==============================================

@st.cache_resource
def load_models():
    # Load your custom text model
    text_model = joblib.load("svm_model1.pkl")
    
    # Load your custom facial emotion model
    facial_emotion_model = load_model("emotion_detection_model.h5")
    
    # Load pre-trained voice model only
    voice_model = Wav2Vec2ForSequenceClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    voice_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    
    return text_model, facial_emotion_model, voice_model, voice_feature_extractor

text_model, facial_emotion_model, voice_model, voice_feature_extractor = load_models()

# ==============================================
# CORE FUNCTIONS
# ==============================================

def predict_emotions(text):
    # Use your custom SVM model
    return text_model.predict([text])[0]

def get_prediction_proba(text):
    # Get probabilities from your SVM model
    proba = text_model.predict_proba([text])[0]
    return dict(zip(text_model.classes_, proba))

def classify_stress(text):
    emotion = predict_emotions(text)
    if emotion in ["happy", "calm", "neutral"]:
        return "No Stress"
    elif emotion in ["sad", "surprise"]:
        return "Mild"
    elif emotion in ["fear", "disgust"]:
        return "Moderate"
    else:
        return "Severe"

def recommend_task(emotion):
    task_rules = {
        "fear": ["Practice deep breathing", "Talk to a mentor for reassurance"],
        "happy": ["Focus on creative tasks", "Collaborate with team"],
        "sad": ["Listen to uplifting music", "Call a friend"],
        "neutral": ["Prioritize important tasks", "Take a mindful break"],
        "angry": ["Take deep breaths", "Go for a walk"],
        "surprise": ["Reflect on what surprised you", "Journal about it"],
        "calm": ["Maintain your routine", "Help others"],
        "disgust": ["Remove yourself from the situation", "Practice grounding techniques"]
    }
    return task_rules.get(emotion, ["No specific recommendation"])

def log_stress(user_id, text, stress_level):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([[user_id, timestamp, text, stress_level]], 
                           columns=["user_id", "timestamp", "text", "stress_level"])
    
    st.session_state.stress_history = pd.concat([st.session_state.stress_history, new_entry], ignore_index=True)
    new_entry.to_csv(STRESS_CSV, mode='a', header=not os.path.exists(STRESS_CSV), index=False)

def analyze_facial_emotion(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, "No face detected"
        
        (x, y, w, h) = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)
        
        # Predict with your custom model
        predictions = facial_emotion_model.predict(roi)[0]
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        dominant_emotion = emotion_labels[np.argmax(predictions)]
        emotion_probs = {emotion: float(prob) for emotion, prob in zip(emotion_labels, predictions)}
        
        return dominant_emotion, emotion_probs
    except Exception as e:
        return None, str(e)

def analyze_speech_emotion(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        inputs = voice_feature_extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = voice_model(**inputs).logits
            predicted_class_id = int(torch.argmax(logits))
            scores = torch.nn.functional.softmax(logits, dim=1)[0].numpy()
        
        emotion_labels = ['angry', 'calm', 'happy', 'sad']
        dominant_emotion = emotion_labels[predicted_class_id]
        emotion_probs = {emotion: float(prob) for emotion, prob in zip(emotion_labels, scores)}
        
        return dominant_emotion, emotion_probs
    except Exception as e:
        return None, str(e)

def add_hr_alert(user_id, alert_type, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    new_alert = pd.DataFrame([[timestamp, user_id, alert_type, message]],
                           columns=["timestamp", "user_id", "alert_type", "message"])
    
    st.session_state.hr_alerts = pd.concat([st.session_state.hr_alerts, new_alert], ignore_index=True)
    new_alert.to_csv(HR_CSV, mode='a', header=not os.path.exists(HR_CSV), index=False)

def check_hr_alerts():
    alerts = pd.concat([
        st.session_state.hr_alerts,
        pd.read_csv(HR_CSV) if os.path.exists(HR_CSV) else pd.DataFrame()
    ]).drop_duplicates()
    
    if alerts.empty:
        st.success("No active HR alerts")
        return
    
    st.subheader("HR Alert Dashboard")
    for _, row in alerts.iterrows():
        if row['alert_type'] == "critical":
            st.markdown(f"""
            <div class="st-alert st-alert-danger">
                <h3>🚨 Critical Alert</h3>
                <p><strong>User:</strong> {row['user_id']}</p>
                <p><strong>Time:</strong> {row['timestamp']}</p>
                <p><strong>Message:</strong> {row['message']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="st-alert st-alert-warning">
                <h3>⚠️ Warning Alert</h3>
                <p><strong>User:</strong> {row['user_id']}</p>
                <p><strong>Time:</strong> {row['timestamp']}</p>
                <p><strong>Message:</strong> {row['message']}</p>
            </div>
            """, unsafe_allow_html=True)

def clear_alerts():
    st.session_state.hr_alerts = pd.DataFrame(columns=["timestamp", "user_id", "alert_type", "message"])
    pd.DataFrame(columns=["timestamp", "user_id", "alert_type", "message"]).to_csv(HR_CSV, index=False)
    st.success("All alerts cleared")

# ==============================================
# STREAMLIT UI COMPONENTS
# ==============================================

def sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
        st.title("Quick Actions")
        
        user_id = st.text_input("Your User ID", key="sidebar_user_id", placeholder="user123")
        
        st.markdown("---")
        
        st.subheader("Quick Analysis")
        quick_text = st.text_area("How are you feeling today?", key="quick_text", height=100)
        
        if st.button("Quick Check"):
            if quick_text.strip():
                with st.spinner("Analyzing..."):
                    emotion = predict_emotions(quick_text)
                    stress = classify_stress(quick_text)
                    probabilities = get_prediction_proba(quick_text)
                    
                    chart = alt.Chart(
                        pd.DataFrame.from_dict(probabilities, orient='index', columns=['probability']).reset_index()
                    ).mark_bar().encode(
                        x='index',
                        y='probability',
                        color=alt.Color('index', scale=alt.Scale(
                            domain=list(emotions_emoji_dict.keys()),
                            range=[v['color'] for v in emotions_emoji_dict.values()]
                        ))
                    ).properties(height=300)
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    emoji = emotions_emoji_dict.get(emotion.lower(), {}).get("emoji", "❓")
                    st.success(f"**Emotion:** {emotion.capitalize()} {emoji}")
                    
                    if stress == "No Stress":
                        st.success(f"**Stress Level:** {stress}")
                    elif stress == "Mild":
                        st.warning(f"**Stress Level:** {stress}")
                    else:
                        st.error(f"**Stress Level:** {stress}")
            else:
                st.warning("Please enter some text to analyze")
        
        st.markdown("---")
        
        st.subheader("Navigation")
        page = st.radio("Go to:", 
                       ["Emotion Detection", 
                        "Stress Analysis", 
                        "My Stress History", 
                        "Live Detection",
                        "HR Alerts"])
        
        st.markdown("---")
        
        if st.button("Check HR Alerts"):
            check_hr_alerts()
    
    return page

def emotion_detection_page():
    st.title("😃 Emotion Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.container():
            st.subheader("Text Analysis")
            raw_text = st.text_area("Express your feelings...", height=150, key="emotion_text")
            
            if st.button("Analyze Emotion"):
                if raw_text.strip():
                    with st.spinner("Detecting emotion..."):
                        prediction = predict_emotions(raw_text)
                        probability = get_prediction_proba(raw_text)
                        
                        emoji_icon = emotions_emoji_dict.get(prediction.lower(), {}).get("emoji", "❓")
                        color = emotions_emoji_dict.get(prediction.lower(), {}).get("color", "#6c757d")
                        
                        st.markdown(f"""
                        <div class="st-emotion-card" style="border-left: 5px solid {color};">
                            <h3>Detected Emotion: {prediction.capitalize()} {emoji_icon}</h3>
                            <p>Confidence: {max(probability.values())*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        proba_df = pd.DataFrame.from_dict(probability, orient='index', columns=['probability'])
                        st.bar_chart(proba_df)
                        
                        st.subheader("Recommended Actions")
                        for task in recommend_task(prediction.lower()):
                            st.markdown(f"""
                            <div class="st-recommendation">
                                <p>✅ {task}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if prediction.lower() in ["anger", "fear"] and max(probability.values()) > 0.8:
                            add_hr_alert(
                                st.session_state.sidebar_user_id,
                                "critical",
                                f"High {prediction} detected in text (confidence: {max(probability.values()):.0%})"
                            )
                else:
                    st.warning("Please enter some text to analyze")

def stress_analysis_page():
    st.title("🧘 Stress Level Analysis")
    
    with st.container():
        st.subheader("Stress Detection")
        user_text = st.text_area("What's on your mind?", height=150, key="stress_text")
        
        if st.button("Analyze Stress Level"):
            if st.session_state.sidebar_user_id.strip() == "":
                st.error("Please enter your User ID in the sidebar first")
            elif user_text.strip():
                with st.spinner("Analyzing stress level..."):
                    stress_level = classify_stress(user_text)
                    
                    alert_class = "st-alert-success" if stress_level == "No Stress" else \
                                "st-alert-warning" if stress_level in ["Mild", "Moderate"] else \
                                "st-alert-danger"
                    
                    stress_messages = {
                        "No Stress": "You seem to be doing well!",
                        "Mild": "Consider taking a short break and practicing deep breathing.",
                        "Moderate": "Try relaxation techniques like meditation or a short walk.",
                        "Severe": "Please consider reaching out for professional support."
                    }
                    
                    st.markdown(f"""
                    <div class="st-alert {alert_class}">
                        <h3>Stress Level: {stress_level}</h3>
                        <p>{stress_messages[stress_level]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    log_stress(st.session_state.sidebar_user_id, user_text, stress_level)
                    
                    if stress_level in ["Moderate", "Severe"]:
                        add_hr_alert(
                            st.session_state.sidebar_user_id,
                            "warning" if stress_level == "Moderate" else "critical",
                            f"{stress_level} stress level detected"
                        )
                    
                    st.subheader("Your Stress History")
                    user_data = st.session_state.stress_history[
                        st.session_state.stress_history["user_id"] == st.session_state.sidebar_user_id
                    ]
                    
                    if not user_data.empty:
                        st.bar_chart(user_data["stress_level"].value_counts())
                    else:
                        st.info("No stress history found for this user.")
            else:
                st.warning("Please enter some text to analyze")

def stress_history_page():
    st.title("📊 My Stress History")
    
    if st.session_state.sidebar_user_id.strip() == "":
        st.warning("Please enter your User ID in the sidebar to view your history")
    else:
        with st.spinner("Loading your history..."):
            user_data = st.session_state.stress_history[
                st.session_state.stress_history["user_id"] == st.session_state.sidebar_user_id
            ]
            
            if user_data.empty:
                st.info("No stress history found. Start by analyzing your stress in the Stress Analysis section.")
            else:
                col1, col2, col3 = st.columns(3)
                total_entries = len(user_data)
                most_common = user_data["stress_level"].value_counts().index[0]
                
                with col1:
                    st.metric("Total Entries", total_entries)
                with col2:
                    st.metric("Most Common Level", most_common)
                with col3:
                    st.metric("Last Analyzed", "Today" if total_entries > 0 else "Never")
                
                st.subheader("Detailed History")
                st.dataframe(user_data.sort_values("timestamp", ascending=False))
                
                st.subheader("Stress Trend Over Time")
                chart = alt.Chart(user_data).mark_line(point=True).encode(
                    x='timestamp:T',
                    y=alt.Y('stress_level:N', sort=['No Stress', 'Mild', 'Moderate', 'Severe']),
                    color='stress_level',
                    tooltip=['timestamp', 'stress_level', 'text']
                ).properties(height=400)
                st.altair_chart(chart, use_container_width=True)

def live_detection_page():
    st.title("🎭 Live Emotion Detection")
    
    tab1, tab2 = st.tabs(["Facial Expression", "Voice Tone"])
    
    with tab1:
        st.subheader("Real-time Facial Emotion Detection")
        
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Instructions
            1. Click 'Start Webcam' to activate your camera
            2. Look at the camera and click 'Capture & Analyze'
            3. View your emotion analysis results
            """)
            
            if st.button("Start Webcam", disabled=st.session_state.webcam_active):
                st.session_state.webcam_active = True
            if st.button("Stop Webcam", disabled=not st.session_state.webcam_active):
                st.session_state.webcam_active = False
        
        with col2:
            if st.session_state.webcam_active:
                img_file_buffer = st.camera_input("Look at the camera...", key="live_webcam")
                if img_file_buffer is not None:
                    with st.spinner("Analyzing your expression..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                            tmp_file.write(img_file_buffer.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            dominant_emotion, emotion_probs = analyze_facial_emotion(tmp_path)
                            
                            if dominant_emotion:
                                emoji_icon = emotions_emoji_dict.get(dominant_emotion.lower(), {}).get("emoji", "❓")
                                color = emotions_emoji_dict.get(dominant_emotion.lower(), {}).get("color", "#6c757d")
                                
                                st.markdown(f"""
                                <div class="st-emotion-card" style="border-left: 5px solid {color};">
                                    <h3>Detected Emotion: {dominant_emotion.capitalize()} {emoji_icon}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                prob_df = pd.DataFrame.from_dict(emotion_probs, orient='index', columns=['Probability'])
                                st.bar_chart(prob_df)
                                
                                st.subheader("Recommended Actions")
                                for task in recommend_task(dominant_emotion.lower()):
                                    st.markdown(f"""
                                    <div class="st-recommendation">
                                        <p>✅ {task}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                if dominant_emotion in ["angry", "fear"]:
                                    add_hr_alert(
                                        st.session_state.sidebar_user_id,
                                        "critical",
                                        f"Concerning facial emotion detected: {dominant_emotion}"
                                    )
                            else:
                                st.error(f"Error analyzing face: {emotion_probs}")
                        
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
                        finally:
                            os.unlink(tmp_path)
    
    with tab2:
        st.subheader("🎙️ Voice Emotion Detection")
        st.info("Record your voice or upload a file for emotion analysis")
        
        # Option selection
        analysis_option = st.radio(
            "Choose input method:",
            ["Record Voice", "Upload Audio File"],
            horizontal=True
        )
        
        if analysis_option == "Record Voice":
            from audio_recorder_streamlit import audio_recorder
            
            # Audio recorder component
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="user",
                icon_size="2x",
            )
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                if st.button("Analyze Recorded Voice"):
                    with st.spinner("Analyzing voice emotion..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_path = tmp_file.name
                        
                        try:
                            dominant_emotion, emotion_probs = analyze_speech_emotion(tmp_path)
                            
                            if dominant_emotion:
                                emoji = emotions_emoji_dict.get(dominant_emotion.lower(), {}).get("emoji", "🎤")
                                color = emotions_emoji_dict.get(dominant_emotion.lower(), {}).get("color", "#6c757d")
                                
                                st.markdown(f"""
                                <div class="st-emotion-card" style="border-left: 5px solid {color};">
                                    <h3>Detected Emotion: {dominant_emotion.capitalize()} {emoji}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                prob_df = pd.DataFrame({
                                    "Emotion": list(emotion_probs.keys()),
                                    "Probability": list(emotion_probs.values())
                                })
                                st.bar_chart(prob_df.set_index("Emotion"))
                                
                                st.subheader("Recommended Actions")
                                for task in recommend_task(dominant_emotion.lower()):
                                    st.markdown(f"""
                                    <div class="st-recommendation">
                                        <p>✅ {task}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                if dominant_emotion in ["angry", "sad"]:
                                    add_hr_alert(
                                        st.session_state.sidebar_user_id,
                                        "warning" if dominant_emotion == "sad" else "critical",
                                        f"Concerning voice emotion detected: {dominant_emotion}"
                                    )
                            else:
                                st.error(f"Error analyzing voice: {emotion_probs}")
                        
                        except Exception as e:
                            st.error(f"Error processing audio: {str(e)}")
                        finally:
                            os.unlink(tmp_path)
        
        else:  # Upload Audio File
            uploaded_file = st.file_uploader("Upload a voice clip (wav only)", type=["wav"])
            
            if uploaded_file is not None:
                st.audio(uploaded_file, format='audio/wav')
                
                if st.button("Analyze Uploaded Voice"):
                    with st.spinner("Analyzing voice emotion..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            dominant_emotion, emotion_probs = analyze_speech_emotion(tmp_path)
                            
                            if dominant_emotion:
                                emoji = emotions_emoji_dict.get(dominant_emotion.lower(), {}).get("emoji", "🎤")
                                color = emotions_emoji_dict.get(dominant_emotion.lower(), {}).get("color", "#6c757d")
                                
                                st.markdown(f"""
                                <div class="st-emotion-card" style="border-left: 5px solid {color};">
                                    <h3>Detected Emotion: {dominant_emotion.capitalize()} {emoji}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                prob_df = pd.DataFrame({
                                    "Emotion": list(emotion_probs.keys()),
                                    "Probability": list(emotion_probs.values())
                                })
                                st.bar_chart(prob_df.set_index("Emotion"))
                                
                                st.subheader("Recommended Actions")
                                for task in recommend_task(dominant_emotion.lower()):
                                    st.markdown(f"""
                                    <div class="st-recommendation">
                                        <p>✅ {task}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                if dominant_emotion in ["angry", "sad"]:
                                    add_hr_alert(
                                        st.session_state.sidebar_user_id,
                                        "warning" if dominant_emotion == "sad" else "critical",
                                        f"Concerning voice emotion detected: {dominant_emotion}"
                                    )
                            else:
                                st.error(f"Error analyzing voice: {emotion_probs}")
                        
                        except Exception as e:
                            st.error(f"Error processing audio: {str(e)}")
                        finally:
                            os.unlink(tmp_path)

def hr_alerts_page():
    st.title("🛎️ HR Alert Dashboard")
    
    check_hr_alerts()
    
    if st.button("Clear All Alerts"):
        clear_alerts()
        st.experimental_rerun()

# ==============================================
# MAIN APP FLOW
# ==============================================

def main():
    page = sidebar()
    
    if page == "Emotion Detection":
        emotion_detection_page()
    elif page == "Stress Analysis":
        stress_analysis_page()
    elif page == "My Stress History":
        stress_history_page()
    elif page == "Live Detection":
        live_detection_page()
    elif page == "HR Alerts":
        hr_alerts_page()

if __name__ == "__main__":
    main()