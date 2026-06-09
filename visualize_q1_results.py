import pandas as pd
import osmnx as ox
import folium

# 1. Load the Scientific Result
print("Loading scientific spanner data...")
df = pd.read_csv("final_scientific_spanner.csv")

# 2. Load map for coordinates
print("Fetching map for visualization...")
G = ox.graph_from_place("Eindhoven, Netherlands", network_type="drive")

# 3. Initialize Map
m = folium.Map(location=[51.4416, 5.4697], zoom_start=13, tiles="cartodbpositron")

# 4. Plot only the Spanner Edges (The Optimized Network)
print(f"Plotting {len(df)} optimized edges...")
for _, row in df.iterrows():
    u, v = int(row['node_u']), int(row['node_v'])
    if u in G.nodes and v in G.nodes:
        line = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
        folium.PolyLine(line, color="blue", weight=2, opacity=0.8).add_to(m)

# 5. Save
m.save("Q1_Scientific_Map.html")
print("✔ SUCCESS! Final Scientific Map saved to Q1_Scientific_Map.html")