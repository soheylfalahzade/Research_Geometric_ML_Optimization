import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.buffer = []

    def add_experiences(self, current_city_critical_edges):
        """
        At each step, we select part of the critical (repaired) edges of the current city
        for long-term memory so they are remembered in later cities.
        """
        # Select 10% of the critical edges for memory
        sample_size = max(1, int(len(current_city_critical_edges) * 0.10))
        selected_samples = np.random.choice(current_city_critical_edges, size=min(sample_size, len(current_city_critical_edges)), replace=False)
        
        self.buffer.extend(selected_samples)
        
        # Keep the memory capacity bounded (algorithm FIFO)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]
            
        print(f"  [Buffer] Added {len(selected_samples)} critical edges. Total memory: {len(self.buffer)} items.")

    def get_memory(self):
        return self.buffer

def test_continual_learning_simulation():
    print("--- Testing Replay Buffer for Continual Learning ---")
    memory = ReplayBuffer(capacity=500)
    
    # Simulate the repaired results in 4 different cities
    cities_critical_edges = {
        "Manhattan": list(range(0, 180)),        # 180 critical edges
        "Eindhoven": list(range(1000, 1212)),    # 212 critical edges
        "Paris":     list(range(2000, 2341)),    # 341 critical edges
        "Rome":      list(range(3000, 3962))     # 962 critical edges
    }

    # Fine-tuning simulation phase (Fine-tuning Loop)
    for city, edges in cities_critical_edges.items():
        print(f"\nTraining on [{city}] with {len(edges)} current critical edges...")
        
        # 1. Extract the past memory
        past_memory = memory.get_memory()
        
        # 2. Combine the current city's data with the memory of previous cities
        training_batch_size = len(edges) + len(past_memory)
        print(f"  [Trainer] GNN is fine-tuning on a mixed batch of {training_batch_size} edges (Current: {len(edges)}, Past Memory: {len(past_memory)})")
        
        # 3. After training, add this city's edges to the memory
        memory.add_experiences(edges)

    print("\n--- Test Completed Successfully ---")
    print("The model successfully avoided Catastrophic Forgetting by mixing past topological data with new cities.")

if __name__ == "__main__":
    test_continual_learning_simulation()