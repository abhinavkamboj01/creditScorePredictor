import json
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import argparse
file_path = "user-wallet-transactions.json"


def load_transactions(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def preprocess_transactions(data):
    rows = []
    for txn in data:
        wallet = txn.get("userWallet")
        action = txn.get("action", "").lower()
        timestamp = int(txn.get("timestamp", 0))
        try:
            raw_amount = float(txn.get("actionData", {}).get("amount", 0))
            # Convert from wei to Ether (assuming 18 decimals)
            amount = raw_amount / (10 ** 18)
        except:
            amount = 0

        rows.append({
            "wallet": wallet,
            "action": action,
            "amount": amount,
            "timestamp": timestamp
        })

    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def engineer_features(df):
    features = []

    for wallet, group in df.groupby("wallet"):
        stats = {"wallet": wallet}
        stats["num_transactions"] = len(group)
        stats["active_days"] = group['datetime'].dt.date.nunique()
        stats["avg_time_between_txns"] = group['timestamp'].diff().mean() if len(group) > 1 else 0

        for action in ["deposit", "borrow", "repay", "redeemunderlying", "liquidationcall"]:
            action_df = group[group["action"] == action]
            stats[f"num_{action}"] = len(action_df)
            stats[f"avg_{action}_amount"] = action_df["amount"].mean() if not action_df.empty else 0
            stats[f"std_{action}_amount"] = action_df["amount"].std() if not action_df.empty else 0

        total_borrow = group[group["action"] == "borrow"]["amount"].sum()
        total_repay = group[group["action"] == "repay"]["amount"].sum()
        stats["borrow_to_repay_ratio"] = total_borrow / total_repay if total_repay > 0 else 1e6
        stats["liquidation_count"] = stats.get("num_liquidationcall", 0)
        stats["net_balance_change"] = (
            group[group["action"] == "deposit"]["amount"].sum()
            - total_borrow
            + total_repay
            - group[group["action"] == "redeemunderlying"]["amount"].sum()
        )

        features.append(stats)

    return pd.DataFrame(features).fillna(0)

def simulate_credit_score(features_df):
    features_df["raw_score"] = (
        features_df["num_deposit"] * 2 +
        features_df["num_repay"] * 2 -
        features_df["num_borrow"] * 1.5 -
        features_df["liquidation_count"] * 10 -
        features_df["borrow_to_repay_ratio"] * 5 +
        features_df["net_balance_change"] * 0.001
    )

    scaler = MinMaxScaler(feature_range=(0, 1000))
    features_df["credit_score"] = scaler.fit_transform(features_df[["raw_score"]])
    return features_df[["wallet", "credit_score"]]

def main(json_file):
    data = load_transactions(json_file)
    df = preprocess_transactions(data)
    features = engineer_features(df)
    scored_wallets = simulate_credit_score(features)
    print(scored_wallets.sort_values(by="credit_score", ascending=False).head(10))
    scored_wallets.to_csv("wallet_credit_scores.csv", index=False)
    print("✅ Scores saved to wallet_credit_scores.csv")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_file = os.path.join(script_dir, "user-wallet-transactions.json")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_file",
        nargs='?',
        default=default_file,
        help="Path to the JSON file with transaction data (default: user-wallet-transactions.json in script directory)"
    )
    args = parser.parse_args()
    main(args.json_file)
