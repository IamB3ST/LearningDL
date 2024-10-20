# Learning DL (updating)

This repository aims to record my process of learning the DL and also I hope this notebook can help others who are interested in DL to know about the it.

> author: Rui Wu

> email: rw761@scarletmail.rutgers.edu


## Computational Resources
| NVIDIA-SMI | Driver | CUDA | GPU | Memory | Pwr |
|:-------|:-------|:-------|:-------|:-------|:-------|
| 552.22 | 552.22 | 12.4 | GeForce RTX 3050Ti | 4096MiB | 95W | 


## About the Notebook

### Modules

This notebook is structured into these parts: `P1 Introduction to Deep Learning`, `P2 Introduction to Large Language Model`
- In `P1 Introduction to Deep Learning`, the classic models/networks like MLP, CNN, ResNet, ViT will be introduced. And train them on the MNIST, FashionMNIST, CIFAR10 and CIFAR100. There are the results we will get in P1:
  
  | model | dataset | accuracy | epoch | augmentation | pre-train |
  |:-------|:-------|:-------|:-------|:-------|:-------|
  | MLP | MNIST | 98.1% | 20 | baseline | no |
  | MLP | FashionMNIST | 81.6% | 20 | baseline | no |
  | AlexNet | FashionMNIST | 91.0% | 30 | baseline | no |
  | ResNet | CIFAR10 | 94.5% | 200 | AutoAugment, RandomErasing | no |
  | ResNet50 | CIFAR100 | 79.1% | 200 | AutoAugment, RandomErasing | yes |

- In `P2 Introduction to Large Language Model`, I will give the code to understand how to handle text data (NLP knowledge), how to use Prompt to make LLM's inference better and other skills for reasoning.

### Content
The notebook is provided in `ipynb` format, compatible with Jupyter Notebook or [Google Colab](https://colab.research.google.com/). Each file contains a comprehensive topic and a corresponding small project for practical understanding.

### Environment
To replicate the environment, follow these simple steps:
```bash
conda create -n ldl python==3.8
conda activate ldl
pip install -r requirements.txt
```

---

## Additional Notes
- For any questions or recommendations regarding this notebook, please create an issue.
- Feel free to reach out to me via email for further discussion or feedback.
- If you'd like to connect, you can find me at Rutgers University.
