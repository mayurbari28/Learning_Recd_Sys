import streamlit as st
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

    st.subheader(f"Recommendations for {selected_user}")
    for r in results:
        st.markdown(
            f"""
            **{r['title']}**  
            • Item ID: `{r['item_id']}`  
            • Category: `{r['category']}`  
            • Difficulty: `{r['difficulty']}`  
            """
        )

st.write("---")
st.caption("Powered by SASRec Transformer Model")
st.caption("Developed by NavGurukul Hackathon Team")
