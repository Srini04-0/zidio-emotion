import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from PIL import Image
import numpy as np

# Custom CSS for dark theme and styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .sidebar .sidebar-content {
        background-color: #1E1E1E;
    }
    .st-bw {
        background-color: #1E1E1E !important;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1E1E1E;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #1E1E1E;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSuccess {
        background-color: #2e7d32 !important;
    }
    .stError {
        background-color: #c62828 !important;
    }
    .stInfo {
        background-color: #1565c0 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #4CAF50 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load models once
@st.cache_resource
def load_text_model():
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    return tokenizer, model

tokenizer, model = load_text_model()

# Title with custom styling
st.markdown(
    """
    <div style='background-color:#1E1E1E;padding:20px;border-radius:10px;margin-bottom:20px;'>
        <h1 style='text-align:center;color:#4CAF50;'>Employee Emotion and Mood Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar with gradient background
with st.sidebar:
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #1E1E1E 0%, #0A0A0A 100%);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    option = st.selectbox(
        "Choose Module", 
        [
            "Text Emotion Detection",
            "Image Emotion Detection",
            "Video Emotion Detection",
            "Task Recommendation",
            "Mood Tracking",
            "Stress Detection",
            "Team Mood Analytics"
        ],
        key='nav_select'
    )

# Helper Functions
def get_text_emotion(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities).item()
    labels = model.config.id2label
    return labels[predicted_label]

def get_face_emotion(uploaded_file):
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = DeepFace.analyze(img_path=img, actions=["emotion"], enforce_detection=False)
    return result[0]["dominant_emotion"]

def recommend_task(emotion):
    recommendations = {
        "happy": "🎨 Engage in creative work or brainstorming",
        "sad": "☕ Take a short break with a warm drink",
        "angry": "🎧 Do focused individual work with calming music",
        "neutral": "📝 Proceed with your daily tasks",
        "fear": "🤝 Talk to a colleague or mentor",
        "disgust": "🧘 Try meditation or solo activities",
        "surprise": "💡 Work on dynamic, innovative tasks"
    }
    return recommendations.get(emotion, "No specific recommendation")

# Mood log file
log_file = "mood_log.csv"
def log_mood(entry):
    df = pd.DataFrame([entry])
    df.to_csv(log_file, mode='a', header=not pd.io.common.file_exists(log_file), index=False)

# Modules with enhanced UI
if option == "Text Emotion Detection":
    st.subheader("📝 Text Emotion Detection")
    with st.container():
        text = st.text_area("Enter your message:", height=150)
        if st.button("🔍 Detect Emotion", key="text_emotion_btn"):
            with st.spinner('Analyzing emotions...'):
                emotion = get_text_emotion(text)
                st.success(f"🎭 Predicted Emotion: {emotion.capitalize()}")
                log_mood({"datetime": datetime.now(), "type": "text", "emotion": emotion})

elif option == "Image Emotion Detection":
    st.subheader("📸 Image Emotion Detection")
    with st.container():
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            cols = st.columns(2)
            with cols[0]:
                st.image(uploaded_file, caption="Uploaded Image", width=300)
            if st.button("😃 Analyze Emotion", key="image_emotion_btn"):
                with st.spinner('Detecting facial expression...'):
                    emotion = get_face_emotion(uploaded_file)
                    cols[1].success(f"🎭 Dominant Emotion: {emotion.capitalize()}")
                    log_mood({"datetime": datetime.now(), "type": "image", "emotion": emotion})

elif option == "Video Emotion Detection":
    st.subheader("🎥 Video Emotion Detection")
    with st.expander("ℹ️ How to use this module"):
        st.info("""
        This feature requires direct webcam access. Please run the following code 
        in your local Python environment for real-time emotion detection.
        """)
    st.code("""
    # Video Emotion Detection Code
    import cv2
    from deepface import DeepFace
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            cv2.putText(frame, result[0]['dominant_emotion'], (50,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        except:
            pass
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    """)

elif option == "Task Recommendation":
    st.subheader("✅ Task Recommendation")
    with st.container():
        mood = st.selectbox(
            "Select your current mood", 
            ["happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"],
            key="mood_select"
        )
        if st.button("💡 Get Recommendation", key="recommend_btn"):
            task = recommend_task(mood)
            st.markdown(f"""
            <div style='background-color:#2E2E2E;padding:20px;border-radius:10px;'>
                <h4 style='color:#4CAF50;'>Recommended Activity:</h4>
                <p style='font-size:18px;'>{task}</p>
            </div>
            """, unsafe_allow_html=True)

elif option == "Mood Tracking":
    st.subheader("📊 Mood Tracking")
    if pd.io.common.file_exists(log_file):
        df = pd.read_csv(log_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        st.dataframe(
            df.tail(10).style.set_properties(**{
                'background-color': '#1E1E1E',
                'color': 'white',
                'border': '1px solid #2E2E2E'
            })
        )
    else:
        st.info("No mood log data found. Start using other modules to collect data.")

elif option == "Stress Detection":
    st.subheader("⚠️ Stress Detection")
    if pd.io.common.file_exists(log_file):
        df = pd.read_csv(log_file)
        negative_emotions = ["sad", "angry", "fear", "disgust"]
        df["stress"] = df["emotion"].isin(negative_emotions)
        stress_count = df["stress"].sum()
        
        if stress_count > 5:
            st.error("🔥 High stress level detected! Consider taking a break or talking to someone.")
        else:
            st.success("🌿 Stress levels are within normal range")
        
        st.metric("Total Stressful Entries", stress_count)
    else:
        st.info("No mood log data available for stress analysis")

elif option == "Team Mood Analytics":
    st.subheader("👥 Team Mood Analytics")
    if pd.io.common.file_exists(log_file):
        df = pd.read_csv(log_file)
        
        st.markdown("### Mood Frequency Distribution")
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        sns.set_style("dark")
        plot = sns.countplot(x="emotion", data=df, palette="viridis", ax=ax)
        
        # Customize plot colors
        plot.spines['bottom'].set_color('white')
        plot.spines['left'].set_color('white')
        plot.tick_params(axis='x', colors='white')
        plot.tick_params(axis='y', colors='white')
        plot.yaxis.label.set_color('white')
        plot.xaxis.label.set_color('white')
        plot.title.set_color('white')
        
        st.pyplot(fig)
    else:
        st.info("No team mood data available yet")