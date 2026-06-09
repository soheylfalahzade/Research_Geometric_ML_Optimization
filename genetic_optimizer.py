import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score

# Load and split dataset once to keep evaluation fast
print("Loading spatial dataset for evolutionary optimization...")
if not os.path.exists("spanner_dataset_pro.csv"):
    print("Error: Dataset not found. Please run data_generator.py first.")
    exit()

df = pd.read_csv("spanner_dataset_pro.csv")
features = ["length", "u_degree", "v_degree", "dx", "dy", "edge_centrality", "u_pagerank", "v_pagerank"]
X = df[features]
y = df["is_spanner_edge"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def calculate_fitness(n_estimators, max_depth, min_samples_split):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        random_state=42,
        class_weight='balanced',
        n_jobs=-1 # Use all CPU cores for speed
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    pruned_recall = recall_score(y_test, y_pred, pos_label=0) # Recall for Pruned Edges (Class 0)
    
    # Custom Balanced Fitness Function
    fitness = 0.5 * auc + 0.5 * pruned_recall
    return fitness, auc, pruned_recall

# Genetic Algorithm Parameters
POP_SIZE = 8
GENERATIONS = 5
MUTATION_RATE = 0.2

# Gene Ranges: n_estimators [10, 200], max_depth [3, 20], min_samples_split [2, 10]
def create_individual():
    return [
        random.randint(10, 200),
        random.randint(3, 20),
        random.randint(2, 10)
    ]

print("Initializing population of chromosomes...")
population = [create_individual() for _ in range(POP_SIZE)]

print(f"Starting Genetic Algorithm optimization across {GENERATIONS} generations...")
best_overall_individual = None
best_overall_fitness = -1
best_overall_auc = 0
best_overall_recall = 0

for gen in range(GENERATIONS):
    print(f"\n--- Generation {gen+1}/{GENERATIONS} ---")
    fitness_results = []
    
    for individual in population:
        n_est, m_depth, min_split = individual
        fit, auc, rec = calculate_fitness(n_est, m_depth, min_split)
        fitness_results.append((fit, individual, auc, rec))
        print(f"Chromosome: [n_est={n_est}, max_depth={m_depth}, min_split={min_split}] -> Fitness: {fit:.4f} (AUC: {auc:.4f}, Recall: {rec:.4f})")
        
        if fit > best_overall_fitness:
            best_overall_fitness = fit
            best_overall_individual = individual
            best_overall_auc = auc
            best_overall_recall = rec
            
    # Selection: Sort by fitness descending
    fitness_results.sort(reverse=True, key=lambda x: x[0])
    
    # Elitisms: Keep top 2 survivors
    new_population = [fitness_results[0][1], fitness_results[1][1]]
    
    # Crossover & Mutation to fill population
    while len(new_population) < POP_SIZE:
        parent1 = random.choice(fitness_results[:4])[1] # Select from top 4 parents
        parent2 = random.choice(fitness_results[:4])[1]
        
        # 1-point crossover
        child = [
            parent1[0], 
            parent2[1], 
            parent2[2]  
        ]
        
        # Mutation
        if random.random() < MUTATION_RATE:
            child[0] = max(10, min(200, child[0] + random.randint(-15, 15)))
        if random.random() < MUTATION_RATE:
            child[1] = max(3, min(20, child[1] + random.randint(-2, 2)))
        if random.random() < MUTATION_RATE:
            child[2] = max(2, min(10, child[2] + random.randint(-1, 1)))
            
        new_population.append(child)
        
    population = new_population

print("\n================== EVOLUTIONARY OPTIMIZATION RESULTS ==================")
print(f"Optimal Hyperparameters Found:")
print(f"-> n_estimators: {best_overall_individual[0]}")
print(f"-> max_depth: {best_overall_individual[1]}")
print(f"-> min_samples_split: {best_overall_individual[2]}")
print(f"Best Fitness Achieved: {best_overall_fitness:.4f}")
print(f"Corresponding ROC-AUC Score: {best_overall_auc:.4f}")
print(f"Corresponding Pruned Edge Recall: {best_overall_recall:.4f}")
print("=======================================================================")