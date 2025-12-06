
import argparse
import sys
import os

# Add path for src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocess import load_and_preprocess
from utils import run_inference


def main(user_id):
    interactions_df, items_df, user_enc, item_enc = load_and_preprocess(
        "data/interactions.csv",
        "data/items.csv"
    )

    results = run_inference(user_id, interactions_df, items_df, user_enc, item_enc)

    print(f"\n=== Recommendations for User {user_id} ===")
    for r in results:
        print(f" • {r['item_id']} → {r['title']} ({r['category']}, {r['difficulty']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning RecSys CLI")
    parser.add_argument("--user", required=True, help="User ID (e.g., U010)")
    args = parser.parse_args()

    main(args.user)
