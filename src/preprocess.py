import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(inter_path, item_path):
    interactions = pd.read_csv(inter_path)
    items = pd.read_csv(item_path)

    # Encode users and items
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    interactions["user"] = user_encoder.fit_transform(interactions["user_id"])
    interactions["item"] = item_encoder.fit_transform(interactions["item_id"])

    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    interactions = interactions.sort_values(["user", "timestamp"])

    return interactions, items, user_encoder, item_encoder
