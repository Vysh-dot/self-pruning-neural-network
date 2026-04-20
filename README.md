 ## Self-Pruning Neural Network (PyTorch)

##  Overview
This project implements a **self-pruning feed-forward neural network** for CIFAR-10 classification using PyTorch.

Unlike traditional pruning (done after training), this model **learns to prune itself during training** by associating each weight with a learnable gate parameter.

---

## Core Idea

Each weight is modified as:

Effective Weight = Weight × Sigmoid(Gate Score)

- Gate values lie between **0 and 1**
- If a gate approaches **0**, the corresponding weight is effectively **removed**
- This enables **dynamic pruning during training**

---

##  Loss Function

The training objective combines accuracy and sparsity:

Total Loss = Classification Loss + λ × Sparsity Loss

Where:
- **Classification Loss** = CrossEntropyLoss  
- **Sparsity Loss** = Sum of all gate values (L1-style penalty)  
- **λ (lambda)** controls the pruning strength  

 Higher λ → more pruning  
 Lower λ → better accuracy  

---

##  Model Architecture

Custom layers: `PrunableLinear`

Network:
- Input Layer: 3 × 32 × 32 (flattened)
- Hidden Layer 1: 512 neurons
- Hidden Layer 2: 256 neurons
- Output Layer: 10 classes

---

##  Dataset

- CIFAR-10 dataset
- Loaded using `torchvision.datasets`

---

##  How to Run

```bash
pip install -r requirements.txt
python self_pruning_net.py

Analysis
The model successfully learned to prune its own connections during training using learnable sigmoid gates and an L1-based sparsity penalty.
As λ increased:
Sparsity increased from 99.57% → 99.95%
Accuracy decreased from 52.64% → 47.96%
This demonstrates the expected trade-off between model compactness and predictive performance.
 Best configuration: λ = 0.0001
It provides the best balance between accuracy and sparsity.
