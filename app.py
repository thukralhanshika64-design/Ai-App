import streamlit as st
from transformers import pipeline

# 1. Title and Description
st.title("🧠 The Vibe Checker")
st.write("Enter any text below, and I'll use AI to tell you if it's **Positive** or **Negative**.")

# 2. Load the AI Model (Cached)
@st.cache_resource
def load_model():
    # We use a standard 'sentiment-analysis' pipeline from Hugging Face.
    # It downloads a small, efficient model (distilbert) by default.
    return pipeline("sentiment-analysis")

# Load the model instantly
with st.spinner("Loading AI Brain..."):
    classifier = load_model()

# 3. User Input
user_text = st.text_area("What's on your mind?", placeholder="Type something like: 'I love coding with Python!'")

# 4. The Button and Result
if st.button("Analyze Vibe"):
    if user_text.strip(): # Check if text is not empty
        # Pass the text to the model
        result = classifier(user_text)
        
        # The model returns a list like [{'label': 'POSITIVE', 'score': 0.99}]
        label = result[0]['label']
        score = result[0]['score']
        
        # Display the result with some dynamic flair
        if label == "POSITIVE":
            st.success(f"**Positive Vibe Detected!** (Confidence: {score:.2f})")
            st.balloons()  # Fun animation
        else:
            st.error(f"**Negative Vibe Detected.** (Confidence: {score:.2f})")
    else:
        st.warning("Please enter some text first!")
