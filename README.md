# Bayesian Deep Learning Notebook (updating)

This repository aims to record my process of learning the Bayesian Deep Learning(BDL) and also I hope this notebook can help others who are interested in BDL to know about the it. The content, structure and pipeline of notebook is refereed to the ["A Survey on Bayesian Deep Learning"](http://www.wanghao.in/paper/CSUR20_BDL.pdf).

> author: Rui Wu

> email: rw761@scarletmail.rutgers.edu

---

## Introduction to BDL

### What is BDL? (one sentence)
It refers to probabilizing the deep neural network as a "perception component", and then unifying it with the probabilistic graphical model as a "task-specific component" under the same probability framework, and end- to-end learning and inference.

### Why use BDL? (one sentence)
Better interpretability, causality, robustness than Deep Learning and higher efficiency with end-to-end study than Probabilistic Graphical Model.

### How to know the recent research about BDL?
You can follow the Github page: [js05212/BayesianDeepLearning-Survey](https://link.zhihu.com/?target=https%3A//github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md)

---

## About the Notebook

### Modules
This notebook is structured into three main modules: Deep Learning, Probabilistic Graphical Models, and Bayesian Deep Learning, following the organization of the ["A Survey on Bayesian Deep Learning"](http://www.wanghao.in/paper/CSUR20_BDL.pdf).

### Content
The notebook is provided in `ipynb` format, compatible with Jupyter Notebook or [Google Colab](https://colab.research.google.com/). Each file contains a comprehensive topic and a corresponding small project for practical understanding.

### Environment
To replicate the environment, follow these simple steps:
```bash
conda create -n bdl python=3.8
conda activate bdl
pip install -r requirements.txt
pip install -e .
```

---

## Additional Notes
- For any questions or recommendations regarding this notebook, please create an issue.
- Feel free to reach out to me via email for further discussion or feedback.
- If you'd like to connect, you can find me at Rutgers University.