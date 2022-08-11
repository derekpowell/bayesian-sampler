import numpyro
numpyro.util.set_host_device_count(4)
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, DiscreteHMCGibbs

from jax import numpy as jnp
from jax import random

import numpy as np
import pandas as pd
import seaborn as sns
import arviz as az

from dfply import *

import pickle

from lib.helpers import *
from lib.models import *
from lib.icc import *


# load data experiment 2

df = load_query_avg_data_exp2() # see data_helpers.py
df = df[df["condition"]!=2] # filter out "warm/snowy" as per paper
df.head()

X_data, y_data = make_model_data(df) # see data_helpers.py
print(len(y_data), "observations")

# fit models

models = {
    "rf": relative_freq,
    "bs_simple": bayesian_sampler_simple,
    "bs": bayesian_sampler_complex,
    "ptn_simple": PTN_simple,
    "ptn": PTN_complex,
    "bs_mlm_simple": bayesian_sampler_simple_mlm_d,
    "bs_mlm": bayesian_sampler_complex_mlm_d,
    "ptn_mlm_simple": PTN_simple_mlm,
    "ptn_mlm": PTN_complex_mlm,
    "ptn_mlm_simplecond": PTN_complex_mlm_simplecond
}

targ_accept_arg = {
    "rf": .80,
    "bs_simple": .80,
    "bs": .80,
    "ptn_simple": .80,
    "ptn": .80,
    "bs_mlm_simple": .95,
    "bs_mlm": .95,
    "ptn_mlm_simple": .95,
    "ptn_mlm": .95,
    "ptn_mlm_simplecond": .95
}

for name in models.keys():
    
    model = models[name]
    kernel = NUTS(model, target_accept_prob=targ_accept_arg[name])
    mcmc = MCMC(kernel, 
               num_warmup=2_000, 
               num_samples=2_000, 
               num_chains=4)

    mcmc.run(random.PRNGKey(0), X_data, y_data)
    
    loo = model_reloo(mcmc, kernel, k_thresh=.70)
    
    az_data = make_arviz_data(mcmc, model, X_data) 
    pickle.dump(az_data, open("local/exp2/az_data_" + name + ".p", "wb"))
    pickle.dump(loo, open("local/exp2/loo_" + name + ".p", "wb"))