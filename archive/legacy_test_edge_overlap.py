import pandas as pd
import numpy as np

# ۱. لود کردن دیتاست لبه‌ها
df = pd.read_csv("final_scientific_spanner.csv")
print(f"Total Edges in Dataset: {len(df)}")

# کاندیداهای هرس (جایی که GNN پیش‌بینی هرس کرده ولی اسپنر لبه واقعی نیست)
candidates = df[df['is_spanner_edge'] == 1].copy()
print(f"Total Spanner Candidate Edges: {len(candidates)}")

# تست دسته ۲۵۰ تایی فعلی برای سنجش میزان تداخل گره‌ها
batch_size = 250
batch = candidates.head(batch_size)

unique_nodes = set(batch['node_u']).union(set(batch['node_v']))
total_nodes_listed = len(batch) * 2

print("\n--- BATCH OVERLAP ANALYSIS ---")
print(f"Batch Size: {batch_size}")
print(f"Total Node References in Batch: {total_nodes_listed}")
print(f"Unique Nodes in Batch: {len(unique_nodes)}")
print(f"Overlap Count: {total_nodes_listed - len(unique_nodes)}")
if (total_nodes_listed - len(unique_nodes)) > 0:
    print("WARNING: Sizable topological overlap detected! This causes the Batch Over-Restoration bug.")
else:
    print("SUCCESS: No overlap detected.")
