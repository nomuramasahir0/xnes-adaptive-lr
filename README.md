## Towards a Principled Learning Rate Adaptation for Natural Evolution Strategies [[slide]](evostar_slide.pdf)

This repository contains the code for the paper "[Towards a Principled Learning Rate Adaptation for Natural Evolution Strategies](https://arxiv.org/abs/2112.10680)".
Please e-mail at masahironomura5325@gmail.com if you have any issue.

### Setup and Run
We use [Poetry](https://python-poetry.org/) for our experimental environment.
You can build the same environment by running the following commands:

```bash
# clone repository
$ git clone git@github.com:nomuramasahir0/xnes_adaptive_lr.git

# build the environment with poetry
$ cd xnes_adaptive_lr
$ poetry install

# run the experiment
$ poetry run python main.py sphere
```


### Citation
If you find this code useful in your research then please cite:

```bibtex
@inproceedings{nomura2022towards,
  title={Towards a Principled Learning Rate Adaptation for Natural Evolution Strategies},
  author={Nomura, Masahiro and Ono, Isao},
  booktitle={International Conference on the Applications of Evolutionary Computation (Part of EvoStar)},
  pages={721--737},
  year={2022},
  organization={Springer}
}
```
