# Comparing probabilistic accounts of probability judgments

Pre-print: https://psyarxiv.com/2bk6f/
OSF: https://osf.io/bpkjf/

## Abstract

Bayesian theories of cognitive science hold that cognition is fundamentally probabilistic, but people’s explicit probability judgments often violate the laws of probability. Two recent proposals, the "Probability Theory plus Noise" (Costello & Watts, 2014) and "Bayesian Sampler" (Zhu et al., 2020) theories of probability judgments, both seek to account for these biases while maintaining that mental credences are fundamentally probabilistic. These models differ in their averaged predictions about people's conditional probability judgments and in their distributional predictions about their overall patterns of judgments. In particular, the Bayesian sampler's Bayesian adjustment process predicts a truncated range of responses as well as a correlation between the average degree of bias and variability trial-to-trial. However, exploring these distributional predictions with participants' raw responses requires a careful treatment of rounding errors and exogenous response processes. Here, I cast these theories into a Bayesian data analysis framework that supports the treatment of these issues along with principled model comparison using information criteria. Comparing the fits of both models on data collected by Zhu and colleagues (2020) I find the data are best explained by an account of biases based on "noise" in the sample-reading process.

## Supplementary materials

This repository contains all supplementary materials for this project. This includes all code to reproduce the analyses reported in the mansucript and the manuscript itself, as well as supplemental analyses.

## Repository organization

- __Manuscript__
  - `paper-rmd/`: folder with reproducible APA-style Rmarkdown document
  - `create-paper-figures.Rmd`: Notebook for translating from python to R for plotting with `reticulate` package
- __Models of models  participant-level query-average data__
  - [`bsampler-numpyro-exp1.ipynb`](https://github.com/derekpowell/bayesian-sampler/blob/main/bsampler-numpyro-exp1.ipynb): Jupyter notebook for fitting models to participant-level query-average data from Experiment 1
  - [`bsampler-numpyro-exp2.ipynb`](https://github.com/derekpowell/bayesian-sampler/blob/main/bsampler-numpyro-exp2.ipynb): Jupyter notebook for fitting models to participant-level query-average data from Experiment 2
  - [`bsampler-model-comp.ipynb`](https://github.com/derekpowell/bayesian-sampler/blob/main/bsampler-model-comp.ipynb): Jupyter notebook for participant-level query-average model comparison
- __Models of trial-level data__
  - [`fit-trial-models.ipynb`](https://github.com/derekpowell/bayesian-sampler/blob/main/fit-trial-models.ipynb): Jupyter notebook for fitting trial-level models. Saved outputs can be downloaded from OSF and put in the `local/` folder. If refitting from scratch strongly recommend using GPU.
- `lib/`: Library folder for python functions
  - `/models.py`: implementations of all models
  - `/simdata.py`: data simulation functions
  - `/icc.py`: functions for reloo
  - `/helpers.py`: data loading and plotting functions
- __Simulation and model validation studies__
  - [`bsampler-numpyro-sim-avgs.ipynb`](https://github.com/derekpowell/bayesian-sampler/blob/main/bsampler-numpyro-sim-avgs.ipynb): simulations and paramter recovery for participant-level query-average models
  - [`fit-trial-models-sim.ipynb`](https://github.com/derekpowell/bayesian-sampler/blob/main/fit-trial-models-sim.ipynb): simulations and parameter recovery for trial-level models
  - [`bsampler-prior-checks.ipynb`](https://github.com/derekpowell/bayesian-sampler/blob/main/bsampler-prior-checks.ipynb): Prior checks for all models
- __Other analyses__
  - [`model-distributional-eda.ipynb`](https://github.com/derekpowell/bayesian-sampler/blob/main/model-distributional-eda.ipynb): Exploratory data analysis of model distributional predictions

### Reproducing the manuscript

1. From https://osf.io/mgcxj/files/ download original data files as .zip and extract into `osfstorage-archive` folder in repo directory.
2. Download saved SVI results for trial-level models from this project's OSF repository and place/unzip in `local/` directory.
3. Use `environment.yml` to create Conda environment.
4. Run fitting notebooks for query-level averaged and trial-level fitting first, then model comparison notebook, and then finally can knit Rmarkdown.
