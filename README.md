# 
<h1 align="center">
Fusion-learning-of-preference-and-bias-from-ratings-and-reviews-for-item-recommendation
</h1>

<p align="center">
  <a href="#2-quick-start🚀">Quick Start</a> •
  <a href="https://www.sciencedirect.com/science/article/pii/S0169023X24000077">Paper</a> •
  <a href="#3-citation☕️">Citation</a>
</p>



Repo for DKE 2024 paper [Fusion learning of preference and bias from ratings and reviews for item recommendation](https://www.sciencedirect.com/science/article/pii/S0169023X24000077).



## 1. Introcution✨
Recommendation methods improve rating prediction performance by learning selection bias phenomenon-users tend to rate items they like. These methods model selection bias by calculating the propensities of ratings, but inaccurate propensity could introduce more noise, fail to model selection bias, and reduce prediction performance. We argue that learning interaction features can effectively model selection bias and improve model performance, as interaction features explain the reason of the trend. Reviews can be used to model interaction features because they have a strong intrinsic correlation with user interests and item interactions. In this study, we propose a preference- and bias-oriented fusion learning model (PBFL) that models the interaction features based on reviews and user preferences to make rating predictions. Our proposal both embeds traditional user preferences in reviews, interactions, and ratings and considers word distribution bias and review quoting to model interaction features. Six real-world datasets are used to demonstrate effectiveness and performance. PBFL achieves an average improvement of 4.46% in root-mean-square error (RMSE) and 3.86% in mean absolute error (MAE) over the best baseline.

## 2. Quick Start🚀

Please download the datasets first.

```sh
conda create -n PBFL_env python=3.8
conda activate PBFL_env
conda pip install -r requitements.txt
python src/PreModel2/run.py
```


## 3. Citation☕️

If you find this repository helpful, please consider citing our paper:

```
@article{LIU2024102283,
title = {Fusion learning of preference and bias from ratings and reviews for item recommendation},
journal = {Data & Knowledge Engineering},
volume = {150},
pages = {102283},
year = {2024},
issn = {0169-023X},
doi = {https://doi.org/10.1016/j.datak.2024.102283},
url = {https://www.sciencedirect.com/science/article/pii/S0169023X24000077},
author = {Junrui Liu and Tong Li and Zhen Yang and Di Wu and Huan Liu},
}
```