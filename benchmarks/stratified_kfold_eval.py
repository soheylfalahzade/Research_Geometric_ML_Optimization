import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from spanner_pipeline import GeometricEdgeSAGE, DEFAULT_DATA_CSV, balanced_loss

N_SPLITS = 5
RANDOM_SEED = 42
EPOCHS = 300


def build_node_features(df, node_map, nodes_unique):
    node_feat_map = {}
    for _, row in df.iterrows():
        node_feat_map[node_map[row["node_u"]]] = [row["u_degree"], row["v_degree"], row["u_pagerank"]]
        node_feat_map[node_map[row["node_v"]]] = [row["v_degree"], row["u_degree"], row["v_pagerank"]]
    x = torch.tensor([node_feat_map.get(i, [0.0, 0.0, 0.0]) for i in range(len(nodes_unique))], dtype=torch.float)
    return x


def run_kfold_evaluation(csv_path=DEFAULT_DATA_CSV):
    print("[K-Fold] Loading dataset...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    features = ["length", "u_degree", "v_degree", "edge_centrality", "u_pagerank", "v_pagerank"]
    df[features] = StandardScaler().fit_transform(df[features])

    labels = df["is_spanner_edge"].values
    print(f"[K-Fold] Total edges: {len(df)}, class balance: {np.bincount(labels.astype(int))}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, labels), start=1):
        print(f"\n[K-Fold] === Fold {fold_idx}/{N_SPLITS} ===")
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        print(f"[K-Fold] Fold {fold_idx}: train class balance {np.bincount(df_train['is_spanner_edge'].astype(int))}, "
              f"test class balance {np.bincount(df_test['is_spanner_edge'].astype(int))}")

        nodes_unique_train = pd.concat([df_train["node_u"], df_train["node_v"]]).unique()
        node_map_train = {n: i for i, n in enumerate(nodes_unique_train)}
        edge_index_train = torch.tensor(
            [[node_map_train.get(u, 0) for u in df_train["node_u"]],
             [node_map_train.get(v, 0) for v in df_train["node_v"]]], dtype=torch.long)
        x_train = build_node_features(df_train, node_map_train, nodes_unique_train)
        y_train = torch.tensor(df_train["is_spanner_edge"].values, dtype=torch.float).view(-1, 1)
        edge_attr_train = torch.tensor(df_train[["length", "edge_centrality"]].values, dtype=torch.float)
        importance_train = torch.tensor(df_train["edge_centrality"].values, dtype=torch.float).view(-1, 1)

        model = GeometricEdgeSAGE().to("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        model.train()
        for epoch in range(1, EPOCHS + 1):
            optimizer.zero_grad()
            out = model(x_train, edge_index_train, edge_attr_train)
            loss = balanced_loss(out, y_train, importance_train)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"  [Fold {fold_idx}] Epoch {epoch:03d} | Loss: {loss.item():.4f}")

        # Evaluate on held-out test fold (nodes may overlap with train since edges share nodes;
        # this is a limitation of edge-level stratified k-fold on a shared graph, noted explicitly)
        nodes_unique_test = pd.concat([df_test["node_u"], df_test["node_v"]]).unique()
        node_map_test = {n: i for i, n in enumerate(nodes_unique_test)}
        edge_index_test = torch.tensor(
            [[node_map_test.get(u, 0) for u in df_test["node_u"]],
             [node_map_test.get(v, 0) for v in df_test["node_v"]]], dtype=torch.long)
        x_test = build_node_features(df_test, node_map_test, nodes_unique_test)
        y_test = df_test["is_spanner_edge"].values
        edge_attr_test = torch.tensor(df_test[["length", "edge_centrality"]].values, dtype=torch.float)

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(x_test, edge_index_test, edge_attr_test)).numpy().flatten()
        preds = (probs > 0.5).astype(int)

        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = float("nan")
        f1 = f1_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)

        print(f"[Fold {fold_idx}] AUC={auc:.4f} F1={f1:.4f} Precision={prec:.4f} Recall={rec:.4f}")

        fold_results.append({
            "fold": fold_idx, "auc": auc, "f1": f1, "precision": prec, "recall": rec,
            "n_train": len(df_train), "n_test": len(df_test)
        })

    results_df = pd.DataFrame(fold_results)
    print("\n" + "=" * 70)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("-" * 70)
    print(f"Mean AUC: {results_df['auc'].mean():.4f} +/- {results_df['auc'].std():.4f}")
    print(f"Mean F1:  {results_df['f1'].mean():.4f} +/- {results_df['f1'].std():.4f}")
    print("=" * 70)

    out_path = os.path.join(os.path.dirname(str(csv_path)), "..", "raw_runs", "stratified_kfold_results.csv")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\n[K-Fold] Results saved to {out_path}")

    return results_df


if __name__ == "__main__":
    run_kfold_evaluation()
