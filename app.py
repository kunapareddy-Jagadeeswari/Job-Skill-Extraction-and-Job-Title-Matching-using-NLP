import os
import json
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import faiss
import streamlit as st
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer, CrossEncoder

st.set_page_config(page_title="Job Skill Extractor", page_icon="🧭", layout="wide")

# ---- Paths: edit if your folders differ ------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Models"
MODEL_PATH = MODEL_DIR / "tuned_linear_svm_model.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
TASKB_DIR = BASE_DIR / "Data" / "TaskB"
TASKB_TRAIN = TASKB_DIR / "training"
CORPUS_PATH = MODEL_DIR / "skills_corpus.pkl"
INDEX_PATH = MODEL_DIR / "skill_faiss.index"

@st.cache_resource(show_spinner="Loading models...")
def load_resources():
    res = {
        "svm": joblib.load(MODEL_PATH),
        "tfidf": joblib.load(TFIDF_PATH),
        "skills_df": pd.read_pickle(CORPUS_PATH),
        "index": faiss.read_index(str(INDEX_PATH)),
        "embedder": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
        "reranker": CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
    }

    # Task B: occupation -> essential / optional skills
    j2s = pd.read_csv(TASKB_TRAIN / "job2skill.tsv", sep="\t", header=None,
                      names=["occupation_uri", "skill_uri", "relation"])
    jobs = json.load(open(TASKB_TRAIN / "jobid2terms.json", encoding="utf-8"))
    skills = json.load(open(TASKB_TRAIN / "skillid2terms.json", encoding="utf-8"))
    first = lambda t: t[0] if isinstance(t, list) and t else t
    j2s["skill_name"] = j2s["skill_uri"].map({k: first(v) for k, v in skills.items()})
    lookup = {}
    for uri, g in j2s.groupby("occupation_uri"):
        lookup[uri] = {
            "title": first(jobs.get(uri)),
            "essential": g.loc[g.relation == "essential", "skill_name"].dropna().tolist(),
            "optional": g.loc[g.relation == "optional", "skill_name"].dropna().tolist(),
        }
    res["occ_lookup"] = lookup
    return res


def predict_occupation(r, t1, t2):
    t1 = (t1 or "").lower().strip()
    t2 = (t2 or t1 or "").lower().strip()

    X = hstack([
        r["tfidf"].transform([t1]),
        r["tfidf"].transform([t2])
    ])

    return r["svm"].predict(X)[0]

def retrieve_and_rerank(r, job_title, candidate_k=50, final_k=10):
    q = r["embedder"].encode([job_title], convert_to_numpy=True,
                             normalize_embeddings=True).astype("float32")
    scores, idx = r["index"].search(q, candidate_k)
    cands = [r["skills_df"].iloc[i] for i in idx[0] if i >= 0]
    rr = r["reranker"].predict([(job_title, c["skill_text"]) for c in cands])
    order = np.argsort(rr)[::-1][:final_k]
    
    # Exclude rerank_score from the final display DataFrame
    return pd.DataFrame([{
        "c_id": cands[i]["c_id"], 
        "skill": cands[i]["skill_text"]
    } for i in order])

def explain_with_groq(job_title, skills_df):
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    key = os.environ.get("GROQ_API_KEY")
    if not key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        key = st.secrets["GROQ_API_KEY"]

    if not key:
        return "Groq API key not found. Set GROQ_API_KEY in your secrets or terminal."
    
    prompt = ChatPromptTemplate.from_template(
        "You are an AI assistant specializing in job skill extraction.\n\n"
        "Job Title: {job_title}\n\nRetrieved Skills:\n{skill_context}\n\n"
        "Instructions:\n"
        "1. Identify the most relevant skills for the job title.\n"
        "2. Explain briefly why they are relevant.\n"
        "3. Use only the provided skills as evidence; do not invent skills.\n"
        "4. Mention that results are predictions and may need human validation.\n\n"
        "Provide a clear, structured explanation."
    )
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=key)
    
    # Context now only passes the Skill ID and Skill text without numerical scores
    ctx = "\n".join(f"Skill ID: {r.c_id}; Skill: {r.skill}"
                    for r in skills_df.itertuples())
                    
    return (prompt | llm | StrOutputParser()).invoke({"job_title": job_title, "skill_context": ctx})


# ------------------------------- UI -----------------------------------------
st.title("🧭 Job Title → Occupation & Skills")
st.caption("TF-IDF + Linear SVM  •  FAISS retrieval  •  Cross-Encoder rerank  •  Groq RAG explanation")

try:
    R = load_resources()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.info("Place the .pkl / .index files next to app.py and the Task B data under Data/TaskB/training.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Skills to show", 5, 20, 10)
    use_llm = st.checkbox("Generate AI explanation (Groq)", value=True)

c1, c2 = st.columns(2)
title1 = c1.text_input("Job title", "Data Scientist")
title2 = c2.text_input("Alternative title (optional)", "Machine Learning Engineer")

if st.button("Analyze", type="primary") and title1.strip():
    uri = predict_occupation(R, title1, title2)
    info = R["occ_lookup"].get(uri)
    st.subheader("Predicted occupation")
    st.write(f"**{info['title'] if info else 'Not found in Task B'}**")
    st.code(uri)

    if info:
        a, b = st.columns(2)
        with a:
            st.markdown(f"**Essential skills ({len(info['essential'])})**")
            st.write(info["essential"][:30])
        with b:
            st.markdown(f"**Optional skills ({len(info['optional'])})**")
            st.write(info["optional"][:30])

    st.subheader("Top skills (FAISS + reranker)")
    with st.spinner("Retrieving..."):
        top = retrieve_and_rerank(R, title1, final_k=top_k)
    st.dataframe(top, use_container_width=True)

    if use_llm:
        st.subheader("AI explanation")
        with st.spinner("Asking Groq..."):
            try:
                st.markdown(explain_with_groq(title1, top))
            except Exception as e:
                st.warning(f"LLM call failed: {e}")