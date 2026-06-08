import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_geometric_optimizer():
    print("Loading real spatial dataset...")
    if not os.path.exists("spanner_dataset_pro.csv") if 'os' in globals() else not pd.io.common.file_exists("spanner_dataset_pro.csv"):
        # Safe import check for OS path
        import os
        if not os.path.exists("spanner_dataset_pro.csv"):
            print("Error: Dataset not found. Please run data_generator.py first.")
            return
        
    df = pd.read_csv("spanner_dataset_pro.csv")
    
    # Define feature matrix (X) and target vector (y)
   # Define advanced feature matrix (X) and target vector (y)
    features = ["length", "u_degree", "v_degree", "dx", "dy", "edge_centrality", "u_pagerank", "v_pagerank"]
    X = df[features]
    y = df["is_spanner_edge"]
    
    # Split into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)} edges, Test set size: {len(X_test)} edges.")
    
    # Train an advanced Random Forest Classifier
    print("Training Random Forest Edge Classifier on real road network...")
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n================== ACADEMIC EVALUATION METRICS ==================")
    print(classification_report(y_test, y_pred, target_names=["Pruned Edge", "Spanner Edge"]))
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")
    print("=================================================================\n")
    
    # Plot and save the research evaluation matrix
    print("Generating research evaluation matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], 
                xticklabels=["Pruned", "Spanner"], yticklabels=["Pruned", "Spanner"])
    axes[0].set_title("Confusion Matrix (Edge Prediction)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    # 2. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)
    axes[1].barh(range(len(indices)), importances[indices], color='skyblue', align='center')
    axes[1].set_yticks(range(len(indices)))
    axes[1].set_yticklabels([features[i] for i in indices])
    axes[1].set_title("Geometric Feature Importance")
    axes[1].set_xlabel("Relative Importance")
    
    plt.tight_layout()
    plt.savefig("final_research_matrix.png", dpi=300)
    print("Evaluation matrices saved to final_research_matrix.png successfully.")

if __name__ == "__main__":
    train_geometric_optimizer()