# MMLRec: Multi-Task and Multi-Scenario Learning Benchmark for Recommendation

## Introduction
MMLRec is the first comprehensive benchmark for multi-task and multi-scenario recommendations. MMLRec implements a wide range of MTL and MSL algorithms,  adopting consistent data processing and data-splitting strategies for fair comparisons. 
We implemented 15 multi-task and multi-scenario methods and evaluated them on five datasets of MTL, five datasets of MSL and two datasets of MTMSL.

## Methods

|             Model             | Paper                                                                                                                                                                                                     |
|:-----------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|         Single-Task:          | Each task is modeled separately, which means that each task is learned using completely independent parameters, with no parameter sharing structure.                                                      |
| MLP (Full shared parameters): | The full parameter sharing structure, meaning that all parametersare shared between different tasks.                                                                                                      |
|         Cross-stitch:         | [Cross-stitch networks for multi-task learning](https://arxiv.org/abs/1604.03539)                                                                                                                         |
|         SharedBottom          | [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/pdf/1706.05098.pdf)                                                                                                        |
|             ESMM              | [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://dl.acm.org/doi/10.1145/3209978.3210104)                                                          |
|             MMoE              | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007)                                                               |
|              PLE              | [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236)                                          |
|              SNR              | [Snr: Sub-network routing for flexible parameter sharing in multi-task learning in e-commerce by exploiting task relationships in the label space](https://ojs.aaai.org/index.php/AAAI/article/view/3788) |
|             MSSM              | [MSSM: A Multiple-level Sparse Sharing Model for Efficient Multi-Task Learning](https://dl.acm.org/doi/10.1145/3404835.3463022)                                                                           |
|             STAR              | [One model to serve all: Star topology adaptive recommender for multi-domain ctr prediction model for efficient multi-task learning](https://arxiv.org/abs/2101.11427)                                    |
|              APG              | [Apg: Adaptive parameter generation network for click-through rate prediction.](https://arxiv.org/abs/2203.16218)                                                                                         |
|             AITM              | [Modeling the sequential dependence among audience multi-step conversions with multi-task learning in targeted display advertising.](https://dl.acm.org/doi/abs/10.1145/3447548.3467071)                  |
|             ESCM              | [ESCM2: entire space counterfactual multi-task model for post-click conversion rate estimation.](https://dl.acm.org/doi/abs/10.1145/3477495.3531972)                                                      |
|             HMoE              | [Improving multi-scenario learning to rank in e-commerce by exploiting task relationships in the label space.](https://dl.acm.org/doi/abs/10.1145/3340531.3412713)                                        |
|            Pepnet             | [Pepnet: Parameter and embedding personalized network for infusing with personalized prior information.](https://dl.acm.org/doi/abs/10.1145/3580305.3599884)                                                                |


## Datasets


Amazon: https://jmcauley.ucsd.edu/data/amazon/

Movielens: https://grouplens.org/datasets/movielens/

Ijcai-2015: https://tianchi.aliyun.com/dataset/42

KuaiRec: https://kuairec.com/

Census-Income: http://archive.ics.uci.edu/dataset/20/census+income

Ijcai-2018: https://tianchi.aliyun.com/dataset/147588

AliExpress: https://tianchi.aliyun.com/dataset/74690

## Requirments

* Python 3.8.13
* Pandas
* tqdm
* sklearn
* numpy
* PyTorch 1.11.0

## Run
Run MTL
```
python main.py --config configs_mtl/config_{dataset_neme}.json
```
Run MSL
```
python main.py --config configs_msl/config_{dataset_neme}.json
```
Run MTMSL
```
python main.py --config configs_mtmsl/config_{dataset_neme}.json
```

