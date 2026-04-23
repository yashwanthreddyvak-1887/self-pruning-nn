# Self-Pruning Neural Network (PyTorch)

## Objective
Build a neural network that learns to prune its own weights during training.

## Key Idea
Each weight is multiplied by a learnable gate (sigmoid).  
If the gate approaches zero, the weight is effectively removed.

## Loss Function
Total Loss = CrossEntropy + λ * L1(gates)

- CrossEntropy: classification accuracy
- L1(gates): encourages sparsity

## Results
- Accuracy: Moderate (due to short training)
- Sparsity: Significant pruning observed

## Observations
- Higher λ → more pruning
- Lower λ → better accuracy

## Tech Used
- PyTorch
- CIFAR-10 dataset
- Matplotlib
