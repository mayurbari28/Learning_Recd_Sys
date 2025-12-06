import streamlit as st
import pandas as pd
import sys
sys.path.append("src")

from preprocess import load_and_preprocess
from utils import run_inference


st.title("🎓 Learning Recommendation System")
st.write("Predict the next best learning item using SASRec")

# Load data
interactions_df, items_df, user_enc, item_enc = load_and_preprocess(
    "data/interactions.csv", "data/items.csv"
)

# User select box
users = sorted(interactions_df["user_id"].unique())
selected_user = st.selectbox("Select User", users)


if st.button("Recommend"):
    results = run_inference(
        selected_user, interactions_df, items_df, user_enc, item_enc, k=5
    )

    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(results)

    st.subheader(f"Recommendations for {selected_user}")

    # Display as table
    st.dataframe(
        df_results[["item_id", "title", "item_type", "category", "difficulty", "format"]],
        use_container_width=True
    )

st.write("---")
st.caption("Powered by SASRec Transformer Model")
st.caption("Developed by NavGurukul Hackathon Team")
