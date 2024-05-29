# Deep Learning Notebook (updating)

This repository aims to record my process of learning the Deep Learning and also I hope this notebook can help others who are interested in Deep Learning to know about the it.

> author: Rui Wu

> email: rw761@scarletmail.rutgers.edu



## About the Notebook

### Modules
This notebook is structured into these main modules: `P1 Introduction to Deep Learning`, `P2 Advanced Methods for Deep Learning`, `P3 Important Tools of Model Training`
- In `P1 Introduction to Deep Learning`, the classic models/networks like MLP, CNN, ResNet, ViT will be introduced. And train them on the MNIST, FashionMNIST, CIFAR10 and CIFAR100. There are the results we will get in P1:
| model | dataset | accuracy | epoch | augmentation | pre-train |
|:-------|:-------|:-------|:-------|:-------|:-------|
| MLP | MNIST | 98.1% | 20 | baseline | no |
| MLP | FashionMNIST | 81.6% | 20 | baseline | no |
| AlexNet | FashionMNIST | 91.0% | 30 | baseline | no |
| ResNet | CIFAR10 | 94.5% | 200 | yes | no |
| WideResNet50 | CIFAR100 | 79.1% | 200 | yes | yes |
|---|---|---|---|---|---|
- In `P2 Advanced Methods for Deep Learning`, we will rethink the relationship between Conv and Attention and how make them better. Then we will introduce the advanced methods about ConvNets family, ViT family or their hybridization, like CoAtNet, ConvNeXt, Swim transformer, TransNeXt and gated method likes MogaNet.

Also there are some track modules you can select after reading main modules: `T1 Introduction to 3D Human`

### Content
The notebook is provided in `ipynb` format, compatible with Jupyter Notebook or [Google Colab](https://colab.research.google.com/). Each file contains a comprehensive topic and a corresponding small project for practical understanding.

### Environment
To replicate the environment, follow these simple steps:
```bash
conda create -n 3dv python=3.8
conda activate 3dv
pip install -r requirements.txt
```

---

## Additional Notes
- For any questions or recommendations regarding this notebook, please create an issue.
- Feel free to reach out to me via email for further discussion or feedback.
- If you'd like to connect, you can find me at Rutgers University.
