import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score

# Load and split dataset
print("Loading spatial dataset for multi-algorithm benchmarking...")
if not os.path.exists("spanner_dataset_pro.csv"):
    print("Error: Dataset not found. Please run data_generator.py first.")
    exit()

df = pd.read_csv("spanner_dataset_pro.csv")
features = ["length", "u_degree", "v_degree", "dx", "dy", "edge_centrality", "u_pagerank", "v_pagerank"]
X = df[features]
y = df["is_spanner_edge"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def calculate_fitness(n_estimators, max_depth, min_samples_split):
    # Ensure parameters are within valid boundaries
    n_estimators = max(10, min(200, int(n_estimators)))
    max_depth = max(3, min(20, int(max_depth)))
    min_samples_split = max(2, min(10, int(min_samples_split)))
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    pruned_recall = recall_score(y_test, y_pred, pos_label=0)
    fitness = 0.5 * auc + 0.5 * pruned_recall
    return fitness

# Common Benchmark Settings to ensure fair comparison
BUDGET = 24  # Total evaluations per algorithm
POP_SIZE = 6
GENERATIONS = 4

# 1. RANDOM SEARCH
def run_random_search():
    print("\nRunning Random Search Baseline...")
    best_fitness = -1
    history = []
    for i in range(BUDGET):
        n_est = random.randint(10, 200)
        m_depth = random.randint(3, 20)
        min_split = random.randint(2, 10)
        fit = calculate_fitness(n_est, m_depth, min_split)
        if fit > best_fitness:
            best_fitness = fit
        history.append(best_fitness)
    return best_fitness, history

# 2. GENETIC ALGORITHM
def run_genetic_algorithm():
    print("\nRunning Genetic Algorithm...")
    population = [[random.randint(10, 200), random.randint(3, 20), random.randint(2, 10)] for _ in range(POP_SIZE)]
    best_fitness = -1
    history = []
    
    for gen in range(GENERATIONS):
        fits = []
        for ind in population:
            fit = calculate_fitness(ind[0], ind[1], ind[2])
            fits.append((fit, ind))
            if fit > best_fitness:
                best_fitness = fit
            history.append(best_fitness)
            
        fits.sort(reverse=True, key=lambda x: x[0])
        new_pop = [fits[0][1], fits[1][1]] # Elitisms
        
        while len(new_pop) < POP_SIZE:
            p1, p2 = fits[0][1], fits[1][1]
            child = [p1[0], p2[1], p2[2]] # Crossover
            if random.random() < 0.2: # Mutation
                child[0] = max(10, min(200, child[0] + random.randint(-15, 15)))
                child[1] = max(3, min(20, child[1] + random.randint(-2, 2)))
            new_pop.append(child)
        population = new_pop
    return best_fitness, history

# 3. DIFFERENTIAL EVOLUTION
def run_differential_evolution():
    print("\nRunning Differential Evolution...")
    population = np.array([[random.randint(10, 200), random.randint(3, 20), random.randint(2, 10)] for _ in range(POP_SIZE)], dtype=float)
    best_fitness = -1
    history = []
    
    # Calculate initial fitness
    fits = [calculate_fitness(ind[0], ind[1], ind[2]) for ind in population]
    for f in fits:
        if f > best_fitness: best_fitness = f
        history.append(best_fitness)
        
    for gen in range(GENERATIONS - 1):
        for i in range(POP_SIZE):
            # Mutation: select 3 random agents different from current
            candidates = [idx for idx in range(POP_SIZE) if idx != i]
            r1, r2, r3 = population[random.sample(candidates, 3)]
            mutant = r1 + 0.5 * (r2 - r3) # F = 0.5
            
            # Crossover
            trial = np.copy(population[i])
            for j in range(3):
                if random.random() < 0.7: # CR = 0.7
                    trial[j] = mutant[j]
                    
            trial_fit = calculate_fitness(trial[0], trial[1], trial[2])
            if trial_fit > fits[i]:
                population[i] = trial
                fits[i] = trial_fit
                if trial_fit > best_fitness:
                    best_fitness = trial_fit
            history.append(best_fitness)
    return best_fitness, history

# 4. PARTICLE SWARM OPTIMIZATION
def run_pso():
    print("\nRunning Particle Swarm Optimization...")
    positions = np.array([[random.randint(10, 200), random.randint(3, 20), random.randint(2, 10)] for _ in range(POP_SIZE)], dtype=float)
    velocities = np.zeros((POP_SIZE, 3))
    pbest = np.copy(positions)
    pbest_fits = [calculate_fitness(p[0], p[1], p[2]) for p in pbest]
    
    gbest_idx = np.argmax(pbest_fits)
    gbest = np.copy(pbest[gbest_idx])
    best_fitness = pbest_fits[gbest_idx]
    
    history = list(np.maximum.accumulate(pbest_fits))
    
    for gen in range(GENERATIONS - 1):
        for i in range(POP_SIZE):
            # Velocity update
            r1, r2 = random.random(), random.random()
            velocities[i] = 0.5 * velocities[i] + 1.5 * r1 * (pbest[i] - positions[i]) + 1.5 * r2 * (gbest - positions[i])
            positions[i] = positions[i] + velocities[i]
            
            # Bound check
            positions[i, 0] = max(10, min(200, positions[i, 0]))
            positions[i, 1] = max(3, min(20, positions[i, 1]))
            positions[i, 2] = max(2, min(10, positions[i, 2]))
            
            fit = calculate_fitness(positions[i, 0], positions[i, 1], positions[i, 2])
            history.append(max(history[-1], fit))
            
            if fit > pbest_fits[i]:
                pbest[i] = positions[i]
                pbest_fits[i] = fit
                if fit > best_fitness:
                    best_fitness = fit
                    gbest = np.copy(positions[i])
    return best_fitness, history

# Run Benchmark
rs_best, rs_hist = run_random_search()
ga_best, ga_hist = run_genetic_algorithm()
de_best, de_hist = run_differential_evolution()
pso_best, pso_hist = run_pso()

print("\n================== BENCHMARK COMPARISON SUMMARY ==================")
print(f"Random Search Best Fitness:        {rs_best:.4f}")
print(f"Genetic Algorithm Best Fitness:    {ga_best:.4f}")
print(f"Differential Evolution Best Fitness: {de_best:.4f}")
print(f"Particle Swarm Best Fitness:       {pso_best:.4f}")
print("==================================================================")

# Plot Convergence Curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, BUDGET + 1), rs_hist, label=f"Random Search (Best: {rs_best:.4f})", linestyle='--', color='gray')
plt.plot(range(1, BUDGET + 1), ga_hist, label=f"Genetic Algorithm (Best: {ga_best:.4f})", color='blue', linewidth=2)
plt.plot(range(1, BUDGET + 1), de_hist, label=f"Differential Evolution (Best: {de_best:.4f})", color='green', linewidth=1.5)
plt.plot(range(1, BUDGET + 1), pso_hist, label=f"Particle Swarm (Best: {pso_best:.4f})", color='red', linewidth=1.5)

plt.title("Metaheuristic Hyperparameter Optimization Convergence Benchmark")
plt.xlabel("Number of Model Evaluations")
plt.ylabel("Balanced Fitness Score")
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig("benchmark_convergence.png", dpi=300)
print("\nBenchmark convergence chart saved to benchmark_convergence.png successfully.")