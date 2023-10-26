<a name="Z8Hwj"></a>
# Notification
This is an implementation of this paper(SR-GNN) based on Tensorflow 2.X, which contains the extra functions as below:

- Log(output figure for loss and MRR@20, P@20)
<a name="i1yiS"></a>
## Requirements
TensorFlow 2.X (version>=2.10 is prefer)<br />Python 3.9<br />CUDA11.6 and above is prefer<br />cudnn8.7.0 and above is prefer<br />**Caution:** For who wants to run in native-Windows, TensorFlow **2.10** was the **last** TensorFlow release that supported GPU on native-Windows.
<a name="rmrbf"></a>
## Paper data and code
This is the code for the AAAI 2019 Paper: [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855). <br />Here are two datasets we used in our paper. After downloaded the datasets, you can put them in the folder datasets/:

- YOOCHOOSE: [http://2015.recsyschallenge.com/challenge.html](http://2015.recsyschallenge.com/challenge.html) or [https://www.kaggle.com/chadgostopp/recsys-challenge-2015](https://www.kaggle.com/chadgostopp/recsys-challenge-2015)
- DIGINETICA: [http://cikm2016.cs.iupui.edu/cikm-cup](http://cikm2016.cs.iupui.edu/cikm-cup) or [https://competitions.codalab.org/competitions/11161](https://competitions.codalab.org/competitions/11161)

Here is a [blog](https://sxkdz.github.io/research/SR-GNN) explaining the paper.
<a name="aRMxw"></a>
## [Citation](https://github.com/CRIPAC-DIG/SR-GNN/tree/e21cfa431f74c25ae6e4ae9261deefe11d1cb488#citation)
```
@inproceedings{Wu:2019ke,
title = {{Session-based Recommendation with Graph Neural Networks}},
author = {Wu, Shu and Tang, Yuyuan and Zhu, Yanqiao and Wang, Liang and Xie, Xing and Tan, Tieniu},
year = 2019,
booktitle = {Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence},
location = {Honolulu, HI, USA},
month = jul,
volume = 33,
number = 1,
series = {AAAI '19},
pages = {346--353},
url = {https://aaai.org/ojs/index.php/AAAI/article/view/3804},
doi = {10.1609/aaai.v33i01.3301346},
editor = {Pascal Van Hentenryck and Zhi-Hua Zhou},
}
```
