# Self-Pruning Neural Network (Simplified)

This project demonstrates a simplified version of a self-pruning neural network.

## Idea
Each weight is multiplied by a gate (sigmoid function).  
If the gate approaches zero, the weight is effectively removed.

## Loss Function
Total Loss = Classification Loss + λ * Sparsity Loss

- Classification Loss: Binary Cross Entropy
- Sparsity Loss: Sum of gate values (L1-like)

## Results
- Accuracy: ~60-70% (synthetic data)
- Sparsity: Shows pruning behavior

## Observations
- Higher lambda increases pruning
- Lower lambda improves accuracy

## Note
This is a simplified prototype demonstrating the concept without heavy frameworks like PyTorch, focusing on core logic.