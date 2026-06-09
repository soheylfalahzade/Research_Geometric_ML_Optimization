import pandas as pd
import osmnx as ox
import folium

# 1. Load the Cleaned Dataset
print("Loading cleaned spanner data...")
df_clean = pd.read_csv("Q1_CLEAN_Spanner.csv")

# 2. Load the Original Graph (Background)
print("Fetching original city layout for background...")
G_orig = ox.graph_from_place("Eindhoven, Netherlands", network_type="drive")

# 3. Initialize Map on Eindhoven center
m = folium.Map(location=[51.4416, 5.4697], zoom_start=14, tiles="cartodbpositron")

# 4. Plot ALL original edges in Light Gray (The "Full" search space)
print("Plotting background layers...")
for u, v, data in G_orig.edges(data=True):
    line = [(G_orig.nodes[u]['y'], G_orig.nodes[u]['x']), (G_orig.nodes[v]['y'], G_orig.nodes[v]['x'])]
    folium.PolyLine(line, color="gray", weight=1, opacity=0.3).add_to(m)

# 5. Plot the CLEAN Spanner in Green (The "Optimized" path)
print(f"Overlaying {len(df_clean)} optimized spanner edges...")
for _, row in df_clean.iterrows():
    u, v = int(row['node_u']), int(row['node_v'])
    if u in G_orig.nodes and v in G_orig.nodes:
        line = [(G_orig.nodes[u]['y'], G_orig.nodes[u]['x']), (G_orig.nodes[v]['y'], G_orig.nodes[v]['x'])]
        folium.PolyLine(line, color="#2ecc71", weight=3, opacity=0.9).add_to(m)

# 6. Save final masterpiece
m.save("Q1_Comparison_Heatmap.html")
print("✔ SUCCESS! Final Comparison Map saved to Q1_Comparison_Heatmap.html")