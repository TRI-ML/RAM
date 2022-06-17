# Object Permanence Emerges in a Random Walk along Memory
A self-supervised approach for learning representations that localize objects under occlusion:
![](figs/preview2.gif)
> [**Object Permanence Emerges in a Random Walk along Memory**](https://arxiv.org/abs/2204.01784),    
> Pavel Tokmakov, Allan Jabri, Jie Li, Adrien Gaidon,   
> *arXiv technical report ([arXiv 2204.01784](https://arxiv.org/pdf/2204.01784.pdf))*


    @inproceedings{tokmakov2022object,
      title={Object Permanence Emerges in a Random Walk along Memory},
      author={Tokmakov, Pavel and Jabri, Allan and Li, Jie and Gaidon, Adrien},
      booktitle={ICML},
      year={2022}
    }

## Abstract
This paper proposes a self-supervised objective for learning representations
that localize objects under occlusion - a property known as object permanence.
A central question is the choice of learning signal in cases of total occlusion.
Rather than directly supervising the locations of invisible objects, we propose a self-supervised objective that requires neither human annotation, nor assumptions about object dynamics. We show that object permanence can emerge by optimizing for temporal coherence of memory: we fit a Markov walk along a space-time graph of memories, where the states in each time step are non-Markovian features from a sequence encoder.
This leads to a memory representation that stores occluded objects and predicts their motion, to better localize them. The resulting model outperforms existing approaches on several datasets of increasing complexity and realism, despite requiring minimal supervision, and hence being broadly applicable.

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Benchmark Evaluation and Training

After [installation](readme/INSTALL.md), follow the instructions in [DATA.md](readme/DATA.md) to setup the datasets. Then check [GETTING_STARTED.md](readme/GETTING_STARTED.md) to reproduce the results in the paper.
We provide scripts for all the experiments in the [experiments](experiments) folder.

## License

RAM is developed upon [PermaTrack](https://github.com/TRI-ML/permatrack) and [CenterTrack](https://github.com/xingyizhou/CenterTrack). Both codebases are released under MIT License themselves. Some code of CenterTrack are from third-parties with different licenses, please check the CenterTrack repo for details. In addition, this repo uses [py-motmetrics](https://github.com/cheind/py-motmetrics), and [TAO codebase](https://github.com/TAO-Dataset/tao) for computing Track AP. ConvGRU implementation is adopted from [this](https://github.com/happyjin/ConvGRU-pytorch) repo. See [NOTICE](NOTICE) for detail. Please note the licenses of each dataset. Most of the datasets we used in this project are under non-commercial licenses.