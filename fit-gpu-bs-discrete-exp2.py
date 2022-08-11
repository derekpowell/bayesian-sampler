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


def bs_dist(p, beta, N):
    return (p * N) / (N + 2 * beta) + beta / (N + 2 * beta)


def bs_dist_inv(x, beta, N):
    return (x - beta / (N + 2. * beta)) * (N + 2. * beta) / N


def bs_dist_cdf(N, beta, a, b, x):
    # where x is untransformed probability
    trans_x = bs_dist_inv(x, beta, N)

    res = jnp.where(
        jnp.logical_or( trans_x <= 0., trans_x >= 1.), 
        jnp.clip(trans_x, 0., 1.), 
        tfp.math.betainc(a, b, jnp.clip(trans_x, 1e-8, 1-1e-8))
    )

    return res


def f_bs(mu, N, beta, responses):
    
    a = mu*N
    b = (1.-mu)*N
    
    n_resps = (responses.shape[0]-1)
    step = int(20/n_resps)
    rnd_unit_scaled = 1/n_resps
    
    lower = jnp.clip((responses/n_resps) - rnd_unit_scaled/2., 1e-8, 1-1e-8)
    upper = jnp.clip((responses/n_resps) + rnd_unit_scaled/2., 1e-8, 1-1e-8)
    
    prob_resps = bs_dist_cdf(N, beta, a, b, upper) - bs_dist_cdf(N, beta, a, b, lower)
    prob_resps = (spread_vec(prob_resps, step) + 1e-30)
    prob_resps = (prob_resps)/jnp.sum(prob_resps)
    
    return(prob_resps)


bs_cat_probs = vmap(f_bs, (0, 0, 0, None))

responses_10 = jnp.linspace(0, 10, num=11)
responses_5 = jnp.linspace(0, 20, num=21)

def bs_complex_mlm_trial_level(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0] 

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
    # k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
    beta_pop = numpyro.sample("beta_pop", dist.Normal(-2.75, .9)) # skewed after sigmoid
    beta_sd = numpyro.sample("beta_sd", dist.HalfCauchy(1))

    N_prime_pop = numpyro.sample("N_prime_pop", dist.Normal(0,2)) # mildly informative
    N_delta_pop = numpyro.sample("N_delta_pop", dist.Normal(0,2)) 
    N_prime_sd = numpyro.sample("N_prime_sd", dist.HalfCauchy(2))
    N_delta_sd = numpyro.sample("N_delta_sd", dist.HalfCauchy(2))
    
    rnd_policy = numpyro.sample("rnd_policy", dist.Dirichlet(jnp.ones(3)))

    # subject-level parameters/priors <--- maybe change to non-centered parameterization for all these
    with numpyro.plate("subj", n_Ps):
        betas = numpyro.sample("beta_r", dist.Normal(0, 1))*beta_sd 
        N_deltas = numpyro.sample("N_delta_r", dist.Normal(0, 1))*N_delta_sd
        N_primes = numpyro.sample("N_prime_r", dist.Normal(0, 1))*N_prime_sd

    # subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

    beta = sigmoid(beta_pop + betas[subj])*10 # constrains beta to [0,10]
    # beta = jnp.exp(beta_pop + betas[subj])
    numpyro.deterministic("beta_subj", jnp.exp(beta_pop + betas))

    # exp() needed to constrain N and N_delta positive
    N = 1 + jnp.exp(N_prime_pop + N_primes[subj]) + jnp.exp(N_delta_pop + N_deltas[subj]) * not_conjdisj # they also required N be at least 1

    numpyro.deterministic("N_subj", 1 + jnp.exp(N_prime_pop + N_primes))
    numpyro.deterministic("N_prime_subj", 1 + jnp.exp(N_prime_pop + N_primes) + jnp.exp(N_delta_pop + N_deltas))

    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]
    
    pi = calc_prob(theta, X_num, X_denom)
    
    numpyro.factor("betacheck", jnp.sum(jnp.where(beta > N, -1e16, 0)))

    # Likelihood
    with numpyro.plate("data", len(trial)):
        
        resp_probs = (
        1./21.*rnd_policy[0] +
        bs_cat_probs(pi, N, beta, responses_5)*rnd_policy[1] + 
        bs_cat_probs(pi, N, beta, responses_10)*rnd_policy[2]
        )
        
        yhat = numpyro.sample("yhat", dist.Categorical(probs=resp_probs), obs=y) # rounded


    return yhat


print("fitting model ...")

kernel = NUTS(bs_complex_mlm_trial_level, target_accept_prob=.80, init_strategy=init_to_median(num_samples=30))

mcmc = MCMC(kernel, 
               num_warmup = 1_500, 
               num_samples = 1_000, 
               num_chains = 2,
               chain_method = "parallel"
              )

mcmc.run(random.PRNGKey(0), X_data, y_data)

print("saving ...")

pickle.dump(mcmc, open("local/bs_mlm_trial_level_disc-exp2-2ksamples.p", "wb"))

az_data = make_arviz_data(mcmc, bs_complex_mlm_trial_level, X_data) # see model_helpers.py
pickle.dump(az_data, open("local/az-bs_mlm_trial_level_disc-exp2-2ksamples.p", "wb"))