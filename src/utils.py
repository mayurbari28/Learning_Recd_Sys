import os
import torch
import pandas as pd
from preprocess import load_and_preprocess
from train_sasrec import train_sasrec
from model_sasrec import SASRec
from inference import recommend

MODEL_PATH = "models/sasrec.pt"
MAX_LEN = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRAIN MODEL IF NEEDED
def train_if_needed(interactions_df):
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("[INFO] Model not found. Training SASRec...")
        train_sasrec(interactions_df)
        print("[SUCCESS] Model trained and saved to models/sasrec.pt")
    else:
        print("[INFO] Model already exists. Skipping training.")



# LOAD MODEL
def load_sasrec_model(interactions_df):
    num_items = interactions_df["item"].nunique()

    model = SASRec(num_items=num_items)
    model.load_state_dict(torch.load("models/sasrec.pt", map_location=device))
    model = model.to(device)
    model.eval()
    return model

# GET USER SEQUENCE
def get_user_sequence(interactions_df, user_id, max_len=50):
    user_rows = interactions_df[interactions_df["user_id"] == user_id]
    if user_rows.empty:
        raise ValueError(f"User {user_id} has no interactions.")
    seq = user_rows.sort_values("timestamp")["item"].tolist()
    return seq[-max_len:]  # last `max_len` interactions



# DECODE ITEM IDS TO ITEM METADATA
def decode_recommendations(pred_ids, item_encoder, items_df):
    decoded_ids = item_encoder.inverse_transform(pred_ids)

    results = []
    for item_id in decoded_ids:
        row = items_df[items_df["item_id"] == item_id].iloc[0]
        results.append({
            "item_id": item_id,
            "title": row["title"],
            "category": row["category"],
            "difficulty": row["difficulty"],
            "item_type": row.get("item_type", "unknown"),
            "format": row.get("format", "unknown")
        })
    return results



# HIGH-LEVEL INFERENCE PIPELINE
def run_inference(user_id, interactions_df, items_df, user_enc, item_enc, k=5):
    # Train model if missing
    train_if_needed(interactions_df)

    # Load model
    model = load_sasrec_model(interactions_df)

    # Extract encoded user sequence
    seq = get_user_sequence(interactions_df, user_id)

    # Run model prediction
    pred_ids = recommend(model, seq, k=k)

    # Decode item IDs → titles
    decoded = decode_recommendations(pred_ids, item_enc, items_df)

    return decoded
