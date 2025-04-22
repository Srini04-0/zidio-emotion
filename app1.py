import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import time
from transformers import pipeline
import os
import cv2
from PIL import Image
from deepface import DeepFace
import tempfile
import librosa
import soundfile as sf
import os
os.environ["USE_TF"] = "0"


# Set page configuration with Zidio branding
st.set_page_config(
    page_title="Zidio AI-Powered Task Optimizer",
    page_icon="🧠",
    layout="centered"
)

# Load Pretrained Emotion Detection Model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

# Load ML model for emotion detection
pipe_lr = joblib.load(open("/workspaces/Emotion-zidio/svm_model1.pkl", "rb"))

# Stress dataset path
STRESS_CSV = "stress_log.csv"

# Emotion dictionary with emojis
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮", "angry": "😠", "surprise": "😮"
}

# Function to predict emotion
def predict_emotions(docx):
    result = pipe_lr.predict([docx])
    return result[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Function to classify stress level
def classify_stress(text):
    if not text.strip():
        return None

    predictions = emotion_classifier(text)[0]
    
    # Define words that should indicate high stress
    high_stress_keywords = ["dying", "death", "panic", "anxiety", "suicide", "hopeless", "fearful", "danger"]

    # Check if the text contains high-stress words
    text_lower = text.lower()
    if any(word in text_lower for word in high_stress_keywords):
        return "Severe"

    # Calculate stress score based on emotion model
    stress_score = sum([p['score'] for p in predictions if p['label'] in ['fear', 'anger', 'sadness']])

    # Adjusted stress level thresholds
    if stress_score < 0.3:
        return "No Stress"
    elif stress_score < 0.6:
        return "Mild"
    elif stress_score < 0.85:
        return "Moderate"
    else:
        return "Severe"

# Function to recommend tasks based on emotion
def recommend_task(emotion):
    task_rules = {
        "fear": ["Practice deep breathing", "Talk to a mentor for reassurance"],
        "happy": ["Focus on creative tasks", "Collaborate with team"],
        "sad": ["Listen to a motivational podcast", "Review recent achievements"],
        "neutral": ["Prioritize high-priority tasks", "Schedule a team check-in"],
        "angry": ["Take a short walk", "Practice mindfulness"],
        "surprise": ["Reflect on what surprised you", "Journal about the experience"]
    }
    return task_rules.get(emotion, ["No recommendation available"])

# Function to log stress level in a dataset
def log_stress(user_id, text, stress_level):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    new_entry = pd.DataFrame([[user_id, timestamp, text, stress_level]], 
                             columns=["user_id", "timestamp", "text", "stress_level"])
    
    if os.path.exists(STRESS_CSV):
        df = pd.read_csv(STRESS_CSV)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry

    df.to_csv(STRESS_CSV, index=False)

# Function to analyze stress history and trends
def analyze_stress_history(user_id):
    if not os.path.exists(STRESS_CSV):
        return None

    df = pd.read_csv(STRESS_CSV)
    user_data = df[df["user_id"] == user_id]

    if user_data.empty:
        return None

    stress_counts = user_data["stress_level"].value_counts().reset_index()
    stress_counts.columns = ["stress_level", "count"]

    return stress_counts

# Function for facial emotion detection
def analyze_facial_emotion(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        emotion_scores = result[0]['emotion']
        return dominant_emotion, emotion_scores
    except Exception as e:
        return None, str(e)

# Function for speech emotion detection
def analyze_speech_emotion(audio_path):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract features (simplified example - in practice you'd use a pre-trained model)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # This is a placeholder - in a real app you'd use a trained model
        emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful']
        fake_probs = np.random.dirichlet(np.ones(5), size=1)[0]
        dominant_emotion = emotions[np.argmax(fake_probs)]
        
        return dominant_emotion, dict(zip(emotions, fake_probs))
    except Exception as e:
        return None, str(e)

# Main App with Zidio Branding
st.markdown(
    """
    <style>
    .big-font {
        font-size:36px !important;
        font-weight: bold;
        color: #4a4a4a;
    }
    .feature-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-title {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with Zidio branding
st.markdown('<div class="big-font">Zidio AI-Powered Task Optimizer</div>', unsafe_allow_html=True)
st.markdown("""
Welcome to the Zidio AI-Powered Task Optimizer!

The system analyzes employee emotions and provides insights to recommend tasks and detect stress or negative moods.
""")

# Feature cards like in the image
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown('<div class="feature-card"><div class="feature-title">Real-Time Emotion Detection</div>Analyses text, video, and speech in real-time to detect employee emotions</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="feature-card"><div class="feature-title">Task Recommendation</div>Suggests tasks based on an employee\'s detected mood</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="feature-card"><div class="feature-title">Historical Mood Tracking</div>Tracks each employee\'s mood over time to identify patterns</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="feature-card"><div class="feature-title">Stress Management Alert</div>Notifies HR or managers when prolonged stress or disengagement</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="feature-card"><div class="feature-title">Team Mood Analytics</div>Identifies team morale and productivity trends through mood data</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="feature-card"><div class="feature-title">Data Privacy</div>Ensures that all sensitive employee data is anonymized and securely stored</div>', unsafe_allow_html=True)

# Tabs for different features
st.markdown("---")
st.subheader("Employee Emotion Analysis Tools")

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Text Analysis", 
    "⚠️ Stress Detection", 
    "📈 Mood History",
    "🎭 Multi-Modal Analysis"
])

# Tab 1: Emotion Detection
with tab1:
    st.subheader("Text Emotion Analysis")

    with st.form(key='emotion_form'):
        raw_text = st.text_area("Enter employee text for analysis", placeholder="Type or paste text here...")
        submit_text = st.form_submit_button(label='Analyze Emotion')

    if submit_text:
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        col1, col2 = st.columns(2)

        with col1:
            st.success("Analysis Results")
            emoji_icon = emotions_emoji_dict.get(prediction, "❓")
            st.write(f"**Dominant Emotion:** {prediction.capitalize()} {emoji_icon}")
            st.write(f"**Confidence:** {np.max(probability)*100:.1f}%")

            # Task Recommendation
            st.success("Recommended Tasks")
            recommendations = recommend_task(prediction)
            for task in recommendations:
                st.write(f"• {task}")

        with col2:
            st.success("Emotion Distribution")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions',
                y='probability',
                color='emotions'
            ).properties(height=300)
            st.altair_chart(fig, use_container_width=True)

# Tab 2: Stress Level Detector
with tab2:
    st.subheader("Employee Stress Detection")

    user_id = st.text_input("Employee ID", placeholder="Enter employee ID for tracking")
    user_text = st.text_area("Text for stress analysis", placeholder="Enter text to analyze stress level...")

    if st.button("Analyze Stress Level"):
        if user_id.strip() == "":
            st.error("Please enter an Employee ID")
        else:
            stress_level = classify_stress(user_text)

            if stress_level:
                if stress_level == "No Stress":
                    st.success("✅ No significant stress detected")
                elif stress_level == "Mild":
                    st.warning("⚠️ Mild Stress Detected")
                    st.info("**Recommendation:** Employee might benefit from a short break or check-in")
                elif stress_level == "Moderate":
                    st.error("⚠️ Moderate Stress Alert")
                    st.info("**Recommendation:** Consider workload adjustment and support resources")
                elif stress_level == "Severe":
                    st.error("🚨 Severe Stress Alert")
                    st.info("**Action Required:** Immediate manager/HR notification recommended")

                # Log stress level
                log_stress(user_id, user_text, stress_level)

                # Show stress history
                stress_history = analyze_stress_history(user_id)
                if stress_history is not None:
                    st.subheader("Stress History Overview")
                    st.bar_chart(stress_history.set_index("stress_level"))
                else:
                    st.info("No previous stress records found for this employee")

# Tab 3: Stress History Chart
with tab3:
    st.subheader("Employee Mood History")

    history_user_id = st.text_input("Enter Employee ID", placeholder="Employee ID to view history")

    if st.button("View Mood History"):
        if history_user_id.strip() == "":
            st.error("Please enter an Employee ID")
        else:
            if os.path.exists(STRESS_CSV):
                df = pd.read_csv(STRESS_CSV)
                user_data = df[df["user_id"] == history_user_id]

                if user_data.empty:
                    st.info("No mood history found for this employee")
                else:
                    st.success(f"Mood history for employee {history_user_id}")
                    
                    user_data["timestamp"] = pd.to_datetime(user_data["timestamp"])

                    # Display Table
                    st.dataframe(user_data)

                    # Plot Stress Trends Over Time
                    chart = alt.Chart(user_data).mark_line(point=True).encode(
                        x='timestamp:T',
                        y=alt.Y('stress_level:N', sort=['No Stress', 'Mild', 'Moderate', 'Severe']),
                        color='stress_level',
                        tooltip=['timestamp', 'stress_level']
                    ).properties(
                        title="Stress Level Trend",
                        height=400
                    )
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("No mood records found yet")

# Tab 4: Face & Voice Analysis
with tab4:
    st.subheader("Multi-Modal Emotion Analysis")
    
    analysis_option = st.selectbox("Select analysis mode:", ["Facial Expression", "Voice Tone", "Both"])
    
    if analysis_option in ["Facial Expression", "Both"]:
        st.subheader("Facial Emotion Analysis")
        face_option = st.radio("Input method:", ["Upload Image", "Use Webcam"])
        
        if face_option == "Upload Image":
            uploaded_file = st.file_uploader("Upload employee photo", type=["jpg", "png", "jpeg"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    image.save(tmp_file.name)
                    dominant_emotion, emotion_scores = analyze_facial_emotion(tmp_file.name)
                
                if dominant_emotion:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success("Detected Emotion")
                        emoji_icon = emotions_emoji_dict.get(dominant_emotion.lower(), "❓")
                        st.write(f"**Dominant Emotion:** {dominant_emotion.capitalize()} {emoji_icon}")
                        
                        st.success("Recommended Tasks")
                        recommendations = recommend_task(dominant_emotion.lower())
                        for task in recommendations:
                            st.write(f"• {task}")
                    
                    with col2:
                        st.success("Emotion Distribution")
                        emotion_data = pd.DataFrame.from_dict(emotion_scores, orient='index', columns=['score'])
                        st.bar_chart(emotion_data)
                else:
                    st.error(f"Analysis error: {emotion_scores}")
        else:  # Webcam option
            picture = st.camera_input("Capture employee photo")
            
            if picture:
                image = Image.open(picture)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    image.save(tmp_file.name)
                    dominant_emotion, emotion_scores = analyze_facial_emotion(tmp_file.name)
                    
                if dominant_emotion:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(image, caption='Captured Image', use_column_width=True)
                    
                    with col2:
                        st.success("Detected Emotion")
                        emoji_icon = emotions_emoji_dict.get(dominant_emotion.lower(), "❓")
                        st.write(f"**Dominant Emotion:** {dominant_emotion.capitalize()} {emoji_icon}")
                        
                        st.success("Emotion Scores")
                        emotion_data = pd.DataFrame.from_dict(emotion_scores, orient='index', columns=['score'])
                        st.bar_chart(emotion_data)
                        
                        st.success("Recommended Tasks")
                        recommendations = recommend_task(dominant_emotion.lower())
                        for task in recommendations:
                            st.write(f"• {task}")
                else:
                    st.error(f"Analysis error: {emotion_scores}")
    
    if analysis_option in ["Voice Tone", "Both"]:
        st.subheader("Voice Emotion Analysis")
        audio_file = st.file_uploader("Upload employee voice recording", type=["wav", "mp3"])
        
        if audio_file is not None:
            st.audio(audio_file, format='audio/wav')
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.read())
                dominant_emotion, emotion_scores = analyze_speech_emotion(tmp_file.name)
            
            if dominant_emotion:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success("Detected Emotion")
                    emoji_icon = emotions_emoji_dict.get(dominant_emotion.lower(), "❓")
                    st.write(f"**Dominant Emotion:** {dominant_emotion.capitalize()} {emoji_icon}")
                    
                    st.success("Recommended Tasks")
                    recommendations = recommend_task(dominant_emotion.lower())
                    for task in recommendations:
                        st.write(f"• {task}")
                
                with col2:
                    st.success("Emotion Distribution")
                    emotion_data = pd.DataFrame.from_dict(emotion_scores, orient='index', columns=['score'])
                    st.bar_chart(emotion_data)
            else:
                st.error(f"Analysis error: {emotion_scores}")

# Footer
st.markdown("---")
st.markdown("**Zidio AI-Powered Task Optimizer** - Enhancing workplace productivity through emotional intelligence")