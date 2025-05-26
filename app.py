import streamlit as st

# Set page config FIRST
st.set_page_config("SmartLearn AI", layout="centered")

from transformers import pipeline
import spacy, random
import docx2txt
import PyPDF2
import tempfile
import wikipedia

# ────────────────────────────────────────────────────────────────
# SmartLearn – Summarizer + Auto‑MCQ Generator (Improved Final)
# ────────────────────────────────────────────────────────────────

# ---------- Cache models (faster loading) ------------------------
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    masker = pipeline("fill-mask", model="roberta-large")
    import spacy

    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

    return summarizer, masker, nlp

summarizer, masker, nlp = load_models()

# ---------- Long text summarizer ---------------------------------
def summarise_long(text: str, chunk_sz=950):
    text = text.replace("\n", " ")
    chunks, cur = [], ""
    for sent in text.split(". "):
        if len(cur) + len(sent) < chunk_sz:
            cur += sent + ". "
        else:
            chunks.append(cur.strip())
            cur = sent + ". "
    if cur:
        chunks.append(cur.strip())

    summary = ""
    for ch in chunks:
        result = summarizer(ch, max_length=180, min_length=60, do_sample=False)[0]
        summary += result["summary_text"] + " "
    return summary.strip()

# ---------- Improved MCQ Generator with Domain-Relevant Distractors ----------
def generate_mcqs(text: str, num_q: int = 10, seed: int = 42):
    random.seed(seed)
    doc = nlp(text)
    sentences = [s for s in doc.sents if len(s.text.split()) > 6]
    random.shuffle(sentences)

    mcqs, used = [], set()

    def get_domain_distractors(answer):
        try:
            summary = wikipedia.summary(answer, sentences=1)
            tokens = [token.text for token in nlp(summary) if token.pos_ in {"NOUN", "PROPN"} and token.is_alpha]
            distractors = list(set(tokens) - {answer})
            return random.sample(distractors, min(10, len(distractors)))
        except:
            return []

    for sent in sentences:
        if len(mcqs) >= num_q:
            break

        target = None
        if sent.ents:
            target = random.choice(sent.ents)
        else:
            nouns = [t for t in sent if t.pos_ in {"NOUN", "PROPN"} and t.is_alpha]
            if nouns:
                target = random.choice(nouns)
        if not target:
            continue

        answer = target.text.strip()
        if answer.lower() in used or len(answer.split()) > 5:
            continue

        distractors = get_domain_distractors(answer)
        distractors = [d for d in distractors if d.lower() != answer.lower() and abs(len(d) - len(answer)) <= 6]
        distractors = list(dict.fromkeys(distractors))  # Remove duplicates

        if len(distractors) >= 3:
            options = distractors[:3] + [answer]
            random.shuffle(options)
            mcqs.append({
                "question": sent.text.replace(answer, "_____"),
                "options": options,
                "answer": answer
            })
            used.add(answer.lower())

    return mcqs

# ---------- File Handling & UI -----------------------------------
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.name.endswith(".docx"):
        return docx2txt.process(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        return None

# ────────────────────────────────────────────────────────────────
# Streamlit App Layout
# ────────────────────────────────────────────────────────────────

st.title("📘 SmartLearn AI – Summarizer + Auto-MCQ Generator")

st.markdown("Upload a **PDF / DOCX / TXT** file or paste raw text:")

option = st.radio("Choose input method:", ["📄 Upload File", "📝 Paste Text"])

raw_text = ""

if option == "📄 Upload File":
    uploaded_file = st.file_uploader("Upload your file", type=["pdf", "docx", "txt"])
    if uploaded_file:
        raw_text = extract_text_from_file(uploaded_file)
elif option == "📝 Paste Text":
    raw_text = st.text_area("Paste your study material here:", height=300)

if raw_text:
    st.markdown("### ✨ Summary")
    with st.spinner("Generating summary..."):
        summary = summarise_long(raw_text)
        st.success("Summary generated!")
        st.write(summary)

    st.markdown("### 🧠 Generate MCQs")
    num_mcqs = st.slider("Number of MCQs to generate", 5, 50, 10)

    with st.spinner("Generating MCQs..."):
        mcqs = generate_mcqs(raw_text, num_q=num_mcqs)
        if mcqs:
            st.success(f"Generated {len(mcqs)} MCQs")
            for i, q in enumerate(mcqs, 1):
                st.markdown(f"**Q{i}. {q['question']}**")
                for opt in q["options"]:
                    st.markdown(f"- {opt}")
                st.markdown(f"<span style='color:green'>✅ Correct: {q['answer']}</span>", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.warning("Could not generate MCQs. Try with more informative input text.")
