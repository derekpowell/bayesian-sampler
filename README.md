# Bayesian Bayesian Sampler

A Bayesian re-analyses of data from:

> Zhu, J.-Q., Sanborn, A. N., & Chater, N. (2020). The Bayesian sampler: Generic Bayesian inference causes incoherence in human probability judgments. _Psychological Review_, 127(5), 719–748.

### Reproducibility

1. From https://osf.io/mgcxj/files/ download original data files as .zip and extract into `osfstorage-archive` folder in repo directory.
2. Analyses are most easily reproduced with the [cogdatasci/jupyter](https://github.com/cogdatasci/jupyter) docker image
3. Some updates and installs will still be necessary, at a terminal in the main repo directory:

```
pip install siuba plotnine
pip install git+https://github.com/arviz-devs/arviz
mkdir local
```
