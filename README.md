# Comparing probabilistic accounts of probability judgments

Pre-print: https://psyarxiv.com/2bk6f/ 
OSF: https://osf.io/bpkjf/

A Bayesian re-analyses of data from:

> Zhu, J.-Q., Sanborn, A. N., & Chater, N. (2020). The Bayesian sampler: Generic Bayesian inference causes incoherence in human probability judgments. _Psychological Review_, 127(5), 719–748.

## Organization

- Manuscript
  - `paper-rmd/`: folder with reproducible APA-style Rmarkdown document
  - `create-paper-figures.Rmd`: Notebook for translating from python to R for plotting with `reticulate` package
- Models of models  participant-level query-average data
  - `fit-exp1...ipynb`: Jupyter notebooks for fitting models to participant-level query-average data
  - `model-comp.ipynb`: Jupyter notebook for participant-level query-average model comparison
- Models of trial-level data
  - `fit-trial-models.ipynb`: Jupyter notebook for fitting trial-level models. Saved outputs can be downloaded from OSF and put in the `local/` folder. If refitting from scratch strongly recommend using GPU.
- `lib/`: Library folder for python functions
  - `/models.py`: implementations of all models
  - `/simdata.py`: data simulation functions
  - `/icc.py`: functions for reloo
  - `/helpers.py`: data loading and plotting functions
- Model validation
  - [`bsampler-numpyro-sim-avgs.ipynb`](https://github.com/derekpowell/bayesian-sampler/blob/main/bsampler-numpyro-sim-avgs.ipynb): simulations and paramter recovery for participant-level query-average models
  - `fit-trial-models-sim.ipynb`: simulations and parameter recovery for trial-level models
  - `bsampler-prior-checks.ipynb`: Prior checks for all models
- Other analyses
  - `model-distributional-eda.ipynb`: Exploratory data analysis of model distributional predictions

### Reproducibility

1. From https://osf.io/mgcxj/files/ download original data files as .zip and extract into `osfstorage-archive` folder in repo directory.
2. Download saved SVI results for trial-level models from this project's OSF repository and place in `local/` directory.
3. Use `environment.yml` to create Conda environment.
4. Run fitting notebooks first, then model comparison notebook, and then finally can knit Rmarkdown.
