# CLIP on low-resource vision

## Overview

This project addresses the **long-tailed distribution** problem in vision-language models, specifically focusing on improving **CLIP** (Contrastive Language-Image Pre-training) performance in **low-resource learning** scenarios. 

The research explores various adaptation techniques to enhance model performance on datasets with imbalanced class distributions.

*The project is part of Trends & Applications of Computer Vision course. MSc in Artificial Intelligence Systems at University of Trento.*

## Key Features

### Adaptation Techniques
1. **[Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685)**
   - Efficiently tunes transformer layers
   - Reduces trainable parameters
   - Preserves model speed and computational efficiency

2. **[Bias-terms Fine-tuning (BitFit)](https://arxiv.org/abs/2106.10199)**
   - Adjusts model bias terms
   - Exposes existing model knowledge
   - Keeps most parameters frozen

3. **[Meta-Adapter](https://arxiv.org/abs/2311.03774)**
   - Facilitates online adaptation with minimal examples
   - Uses cross-attention for feature alignment
   - Implements meta-learning approach

4. **[Label Preserving & Breaking Data Augmentation](https://arxiv.org/abs/2401.04716)**
   - Generates augmented training data using Stable Diffusion
   - Creates label-preserving and label-breaking images
   - Introduces diversity while maintaining semantic integrity

## Experimental Datasets

1. [EuroSAT](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
2. [Circuits-diagrams](https://uvaauas.figshare.com/articles/dataset/Low-Resource_Image_Transfer_Evaluation_Benchmark/25577145)


## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/low-resource-vision-clip.git

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Example training command
python main.py --dataset circuits 
               --root_path data/circuits/
               --shots 16 
               --enable_lora 
               --enable_BitFit 
               --enable_MetaAdapter
               --enable_breaking_loss
```

## Key Arguments

- `--dataset`: Choose dataset (e.g., 'eurosat', 'circuits')
- `--root_path`: Path of the data
- `--shots`: Number of few-shot examples
- `--backbone`: CLIP model backbone (default: 'ViT-B/16')
- `--enable_lora`: Enable Low-Rank Adaptation
- `--enable_BitFit`: Enable Bias-terms Fine-tuning
- `--enable_MetaAdapter`: Enable Meta-Adapter
- `--enable_breaking_loss`: Enable Breaking Loss

## Citation

If you use this work in your research, please cite:

```bibtex
@article{LorenziCazzolaFacchini2024LowResourceVision,
  title={CLIP on Low-Resource Vision},
  author={Lorenzi, Alessandro and Cazzola, Luca and Facchini, Omar},
  year={2024},
  institution={University of Trento}
}
```


## Authors

- Alessandro Lorenzi
- Luca Cazzola
- Omar Facchini








