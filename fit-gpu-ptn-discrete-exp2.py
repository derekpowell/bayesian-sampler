import numpyro
numpyro.enable_x64()
numpyro.set_platform('gpu')

import jax
print(jax.devices())

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median, init_to_value, init_to_sample


import tensorflow_probability.substrates.jax as tfp

from jax import numpy as jnp
from jax import random, vmap


import numpy as np
import pandas as pd
import seaborn as sns
import arviz as az

import siuba as s
from siuba import _

from dfply import *

from plotnine import *

import pickle

## --- run local scripts
from model_helpers import *
from models import *

## -- load data

def load_data_exp2_trials():
    df = load_raw_data(2)
    
    df["condition"] = np.select(
    [
        df.querydetail.str.contains("windy|cloudy"), 
        df.querydetail.str.contains("cold|rainy"),
        df.querydetail.str.contains("warm|snowy")

    ], 
    [
        0,
        1,
        2
    ], 
    default=0 )
    
    original_ids = list(np.unique(df.ID))
    fix_id_dict = {original_ids[i]:i for i in range(0, len(original_ids))}
    
    df = df.assign(ID = df.ID.apply(lambda x: fix_id_dict[x]))

    return df

df = load_data_exp2_trials() # see data_helpers.py
df = df[df["condition"]!=2] # filter out "warm/snowy" as per paper
# df = df.sort_values(by=["ID","block","condition"]) # don't think I need to sort?

df = (df >> 
      s.mutate(block = _.block-1) >> 
#       s.filter(_.block==0, _.ID < 20) >>
      # s.group_by(_.ID, _.condition, _.querytype, _.querydetail) >> 
      # s.summarize(estimate = _.estimate.mean()) >>
      s.mutate(estimate =  np.round(_.estimate*20))
     )
df.estimate = df.estimate.astype("int64")

X_data, y_data = make_model_data(df) # see data_helpers.py
print(len(y_data), "observations")

## --- 

def prob_judge_BS(theta, X_num, X_denom, N, beta):

    pi = calc_prob(theta, X_num, X_denom)
    p_bs = pi * N / (N + 2.*beta) + beta/(N + 2.*beta)
    
    return p_bs

def prob_judge_BS_d(theta, X_num, X_denom, d):

    pi = calc_prob(theta, X_num, X_denom)
    return calc_ptn_prob(pi, d)


## rounding stuff
from jax import vmap

def spread_vec(x, step_size): # this works without static arguments
    base_steps = x.shape[0]
    x_split = jnp.split(x, base_steps)
    pad = jnp.zeros(step_size-1)
    probs = jnp.stack([jnp.concatenate((i,pad)) for i in x_split]).flatten()
    probs = probs[0:21]
    return probs/jnp.sum(probs)


def f(mu, k, responses):
    
    a = mu*k
    b = (1.-mu)*k
    
    n_resps = (responses.shape[0]-1)
    step = int(20/n_resps)
    rnd_unit_scaled = 1/n_resps
    
    lower = jnp.clip((responses/n_resps) - rnd_unit_scaled/2., 1e-8, 1-1e-8)
    upper = jnp.clip((responses/n_resps) + rnd_unit_scaled/2., 1e-8, 1-1e-8)
    
    prob_resps = tfp.math.betainc(a, b, upper) - tfp.math.betainc(a, b, lower)
    prob_resps = (spread_vec(prob_resps, step) + 1e-30)
    prob_resps = (prob_resps)/jnp.sum(prob_resps)
    # prob_resps = (prob_resps + 1e-30) / jnp.sum(prob_resps) # add err to prevent divergences
    
    return(prob_resps)


lbeta_cat_probs = vmap(f, (0, None, None)) # change to map for k

responses_10 = jnp.linspace(0, 10, num=11)
responses_5 = jnp.linspace(0, 20, num=21)

def ptn_simplecond_mlm_trial_level_disc(data, y=None):
    
    # parameterized in terms of d and d' for comparison of model fit

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0] 
    
    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
    rnd_policy = numpyro.sample("rnd_policy", dist.Dirichlet(jnp.ones(3)))
    
    d_base_pop = numpyro.sample("d_base_pop", dist.Normal(-1.0, 1.0))
    d_delta_pop = numpyro.sample("d_delta_pop", dist.Normal(0, .5)) # bias toward lower values for non conj/disj trials
    d_base_sd = numpyro.sample("d_base_sd", dist.LogNormal(-1., 1.)) # was halfcauchy(1)
    d_delta_sd = numpyro.sample("d_delta_sd", dist.LogNormal(-1., 1.)) # approx uniform altogether we hope

    # subject-level parameters/priors 
    with numpyro.plate("subj", n_Ps):
        d_bases = numpyro.sample("d_base_r", dist.Normal(0, 1))
        d_deltas = numpyro.sample("d_delta_r", dist.Normal(0, 1))
#         ks = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
        
    # subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))
    
    d_lin = (d_base_pop + 
             d_bases[subj]*d_base_sd + 
             jnp.exp(d_delta_pop + d_delta_sd*d_deltas[subj])*conjdisj
            )  # exp() constrains d_delta to be positive
    d = sigmoid(d_lin)/2.0 # require this be in [0, 1/3]
    
    numpyro.deterministic("d_subj", sigmoid(d_base_pop + d_bases*d_base_sd)/3.)
    numpyro.deterministic("d_prime_subj", 
                          sigmoid(d_base_pop + 
                                  d_bases*d_base_sd + 
                                  jnp.exp(d_delta_pop + d_deltas*d_delta_sd)
                                 )/2.
                         )
    
    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]
        
    p_bs = prob_judge_BS_d(theta, X_num, X_denom, d)
#     k = ks[subj]
    
    resp_probs = (
        1./21.*rnd_policy[0] +
        lbeta_cat_probs(p_bs, k, responses_5)*rnd_policy[1] + 
        lbeta_cat_probs(p_bs, k, responses_10)*rnd_policy[2]
    )

    # Likelihood
    with numpyro.plate("data", len(trial)):
        # yhat = numpyro.sample("yhat", dist.Beta(p_bs*k, (1-p_bs)*k), obs=y)
        yhat = numpyro.sample("yhat", dist.Categorical(probs=resp_probs), obs=y) # rounded
        return(yhat)

        
print("fitting model ...")

kernel = NUTS(ptn_simplecond_mlm_trial_level_disc, target_accept_prob=.80, init_strategy=init_to_median(num_samples=30))
# kernel_bs = NUTS(mymodel, target_accept_prob=.80, init_strategy=init_to_sample())

mcmc = MCMC(kernel, 
               num_warmup = 1_000, 
               num_samples = 1_000, 
               num_chains = 2,
               chain_method = "sequential"
              )

mcmc.run(random.PRNGKey(0), X_data, y_data)

print("saving ...")

pickle.dump(mcmc, open("local/ptn_simplecond_mlm_trial_level_disc-exp2-2ksamples.p", "wb"))

az_data = make_arviz_data(mcmc, ptn_simplecond_mlm_trial_level_disc, X_data) # see model_helpers.py
pickle.dump(az_data, open("local/az-ptn_simplecond_mlm_trial_level_disc-exp2-2ksamples.p", "wb"))