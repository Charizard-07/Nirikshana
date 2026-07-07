import streamlit as st
from src.predict import predict

st.set_page_config(page_title="Fake vs Real News Detector")
st.title("Fake vs Real News Detector")
st.write("Paste a news headline or article below to check if it's likely real or fake.")

text = st.text_area("News text:", height=200, placeholder="Paste headline or article here...")

if st.button("Analyze") and text:
    with st.spinner("Analyzing..."):
        result = predict(text)

    label = result["label"]
    confidence = result["confidence"]

    if label == "Real":
        st.success(f"Predicted: **Real** (confidence: {confidence:.1%})")
    else:
        st.error(f"Predicted: **Fake** (confidence: {confidence:.1%})")

    st.caption("Note: this model detects stylistic patterns typical of fake vs. real news writing — it does not fact-check claims.")