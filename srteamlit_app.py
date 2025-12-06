import streamlit as st
import pandas as pd
import sys
sys.path.append("src")

from preprocess import load_and_preprocess
from utils import run_inference
from recomd_rag import recomd_rag

# STREAMLIT APP
st.title("🎓 Learning Recommendation System")
st.write("Choose between RAG (LLM-based) and SASRec (sequence-based).")

# Load base data
interactions_df, items_df, user_enc, item_enc = load_and_preprocess(
    "data/interactions.csv", "data/items.csv"
)

users = sorted(interactions_df["user_id"].unique())
selected_user = st.selectbox("Select a User", users)

# RAG/SASRec Toggle Switch
mode = st.radio("Recommendation Mode", ["RAG", "SASRec"], horizontal=True)

# API Key (only needed for RAG)
api_key = None
if mode == "RAG":
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Button to run recommendation
if st.button("Get Recommendations"):
    st.subheader(f"Recommendations for {selected_user}")

    #RAG MODE
    if mode == "RAG":
        if not api_key:
            st.error("Please enter the OpenAI API key to use RAG.")
        else:
            with st.spinner("Running RAG (Semantic Search + LLM reasoning)..."):
                df, reasons = recomd_rag.recommend_rag(selected_user, api_key)

            if df.empty:
                st.error("No recommendations found.")
            else:
                st.dataframe(df, use_container_width=True)

                if reasons:
                    st.write("### 🔍 LLM Explanation")
                    for r in reasons:
                        st.write(f"- **{r['item_id']}** → {r['reason']}")

    #SASRec MODE
    else:
        with st.spinner("Running SASRec model (sequence-based)..."):
            df = run_inference(
        selected_user, interactions_df, items_df, user_enc, item_enc, k=5
    )

        if df.empty:
            st.warning("User does not have enough interaction history for SASRec.")
        else:
            st.dataframe(df, use_container_width=True)

st.write("---")
st.caption("Powered by SASRec Transformer Model")
st.caption("Developed by NavGurukul Hackathon Team")
