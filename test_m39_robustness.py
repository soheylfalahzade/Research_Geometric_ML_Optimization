import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv

# ۱. مدل ارتقا یافته با MC-Dropout
class RobustGNN(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_idx):
        h = F.relu(self.conv1(x, edge_idx))
        h = self.dropout(h) # لایه Dropout برای نمونه‌گیری آماری
        return torch.sigmoid(self.classifier(h))

# ۲. تست بصری Uncertainty
def visualize_uncertainty():
    model = RobustGNN()
    model.train() # بسیار مهم: مدل در حالت train می‌ماند تا dropout فعال باشد
    
    x = torch.randn(100, 2)
    edge_idx = torch.randint(0, 100, (2, 200))
    
    # نمونه‌گیری مونت‌کارلو (MC-Dropout)
    samples = torch.stack([model(x, edge_idx).flatten() for _ in range(50)])
    mean = samples.mean(dim=0).detach().numpy()
    std = samples.std(dim=0).detach().numpy()
    
    # رسم نمودار بصری برای داور
    plt.figure(figsize=(8, 4))
    plt.hist(std, bins=20, color='purple', alpha=0.7)
    plt.title("Distribution of Model Uncertainty (MC-Dropout)")
    plt.xlabel("Standard Deviation (Uncertainty)")
    plt.ylabel("Frequency")
    plt.savefig("uncertainty_dist.png")
    print("✓ Uncertainty Histogram saved as 'uncertainty_dist.png'")
    print(f"Mean Uncertainty: {std.mean():.4f}")

# ۳. تست Replay Buffer (شکاف ۳)
class ContinualTrainer:
    def __init__(self):
        self.buffer = []
        
    def update(self, new_data):
        self.buffer.extend(new_data)
        if len(self.buffer) > 200: self.buffer = self.buffer[-200:]
        return len(self.buffer)

# اجرای تست
visualize_uncertainty()
trainer = ContinualTrainer()
print(f"Replay Buffer Capacity Test: {trainer.update([1]*50)} items stored.")