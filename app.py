import streamlit as st
import joblib
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

st.set_page_config(page_title="Spam Classifier", page_icon="📧")

@st.cache_resource
def load_model():
    model      = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(w for w in text.split() if w not in stop_words)
    return text

def predict(text):
    cleaned    = preprocess(text)
    vectorized = vectorizer.transform([cleaned])
    pred       = model.predict(vectorized)[0]
    # SVM doesn't support predict_proba — use decision score instead
    score      = model.decision_function(vectorized)[0]
    spam_conf  = round(min(max((score + 2) / 4 * 100, 0), 100), 2)
    ham_conf   = round(100 - spam_conf, 2)
    return pred, ham_conf, spam_conf

# ── UI ───────────────────────────────────────────────────────
st.title("📧 Spam Email Classifier")
st.markdown("Paste any message below to check if it's **spam or not**.")
st.divider()

st.markdown("**💡 Try an example:**")
col1, col2 = st.columns(2)
spam_example = "Congratulations! You've won a $1,000 Walmart gift card. Click here NOW to claim!"
ham_example  = "Hey, are we still on for lunch tomorrow? Let me know what time works."
if col1.button("🚨 Spam Example"):
    st.session_state['input_text'] = spam_example
if col2.button("✅ Ham Example"):
    st.session_state['input_text'] = ham_example

input_text = st.text_area(
    "Enter your message:",
    value=st.session_state.get('input_text', ''),
    height=180,
    placeholder="Type or paste a message here..."
)

if st.button("Classify Message", use_container_width=True, type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        pred, ham_conf, spam_conf = predict(input_text)
        st.divider()
        if pred == 1:
            st.error(f"## SPAM Detected! ({spam_conf}% confidence)")
        else:
            st.success(f"## Not Spam! ({ham_conf}% confidence)")

        st.divider()
        st.subheader("Confidence Breakdown")
        c1, c2 = st.columns(2)
        c1.metric(" Ham",  f"{ham_conf}%")
        c2.metric("Spam", f"{spam_conf}%")
        st.progress(spam_conf / 100)

st.divider()
st.caption("Built with SVM + Streamlit | UCI SMS Spam Dataset")


