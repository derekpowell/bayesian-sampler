import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import numpy as np

from jax.nn import softplus
import tensorflow_probability.substrates.jax as tfp

from lib.helpers import *

## -------------------------------------
## Relative Frequency
## -------------------------------------

def calc_prob(theta, X_num, X_denom):

    numerator = jnp.sum(theta*X_num, axis=-1)
    denom = jnp.sum(theta*X_denom, axis=-1)

    return jnp.divide(numerator, denom)

def relative_freq(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0]

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])

    # population level parameters/priors
#     sigma = numpyro.sample("sigma", dist.HalfCauchy(.1)) # for Normal() response noise version
    k = numpyro.sample("k", dist.HalfNormal(50)) # noise parameter

    # need a theta per person-query
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]

    pi = calc_prob(theta, X_num, X_denom)

    with numpyro.plate("data", len(trial)):
#         yhat = numpyro.sample("yhat", dist.Normal(pi, sigma), obs = y) # wrong but replicates paper
        yhat = numpyro.sample("yhat", dist.Beta(pi*k, (1-pi)*k), obs=y)

    return yhat

## -------------------------------------
## Bayesian Sampler
## -------------------------------------

def prob_judge_BS(theta, X_num, X_denom, N, beta):

    pi = calc_prob(theta, X_num, X_denom)
    p_bs = pi * N / (N + 2.*beta) + beta/(N + 2.*beta)

    return p_bs

def prob_judge_BS_d(theta, X_num, X_denom, d):

    pi = calc_prob(theta, X_num, X_denom)
    return calc_ptn_prob(pi, d)


def bayesian_sampler_simple(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0]

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    not_conjdisj = abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfNormal(50)) # noise parameter

    # subject-level parameters/priors
#    with numpyro.plate("subj", n_Ps):
#        betas = numpyro.sample("beta", dist.Uniform(0.0, 1.0)) # bounded to [0,1] as in paper
#        Ns = numpyro.sample("N_prime", dist.Normal(0,3))

    with numpyro.plate("subj", n_Ps):
#        ds = numpyro.sample("d_base", dist.Beta(1, 1))
        ds = numpyro.sample("d_base", dist.Normal(0, 1.75))

    # subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

#    beta = betas[subj] # use jnp.exp() if unbounded
    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]

    d_lin = ds[subj]
    d = sigmoid(d_lin)/3.
#    N_lin = Ns[subj] #  < --------------
#    N = 1 + jnp.exp(N_lin) # they required N be at least 1

    p_bs = prob_judge_BS_d(theta, X_num, X_denom, d)

    # Likelihood
    with numpyro.plate("data", len(trial)):
        yhat = numpyro.sample("yhat", dist.Beta(p_bs*k, (1-p_bs)*k), obs=y)

    return yhat


def bayesian_sampler_complex(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0]

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfNormal(50)) # noise parameter

    # subject-level parameters/priors
    with numpyro.plate("subj", n_Ps):
#        betas = numpyro.sample("beta", dist.Uniform(0.0, 1.0)) # bounded to [0,1] as in paper
#        N_deltas = numpyro.sample("N_delta", dist.Normal(0,3)) # 3/25/21, 11:10 PM changed this
#        N_primes = numpyro.sample("N_prime", dist.Normal(0,3))
        d_bases = numpyro.sample("d_base", dist.Normal(0, 1))
        d_deltas = numpyro.sample("d_delta", dist.Normal(0, 1)) # sum of these is approx uniform

    # subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

#    beta = betas[subj] # use jnp.exp() if unbounded
    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]
#     N = 1 + N_primes[subj] + N_deltas[subj] * not_conjdisj
#    N_lin = N_primes[subj] + N_deltas[subj] * not_conjdisj #  < -------------- gah, fix this 3/26/21, 9:41 AM
#    N = 1 + jnp.exp(N_lin) # they required N be at least 1
    d_lin = d_bases[subj] + jnp.exp(d_deltas[subj]) * conjdisj # constrain d < d'
    d = sigmoid(d_lin)/3. # require this be below .33

    p_bs = prob_judge_BS_d(theta, X_num, X_denom, d)

    # Likelihood
    with numpyro.plate("data", len(trial)):
        yhat = numpyro.sample("yhat", dist.Beta(p_bs*k, (1-p_bs)*k), obs=y)

    return yhat

## -------------------------------------
## PT+N
## -------------------------------------

def prob_judge_PTN(trial, theta, X_num, X_denom, X_A, d):

    conditional = jnp.array([is_cond(i) for i in trial])
    not_conditional = abs(1-conditional)

    p_numerator = jnp.sum(theta*X_num, axis=-1) #  = P(A & B)
    pA = jnp.sum(theta*X_A, axis=-1)
    num_or_denom = jnp.logical_or(X_A, X_denom).astype("float32")
    p_num_or_denom = jnp.sum(theta * num_or_denom , axis=-1) # = P(A or B)


    p_denom = jnp.sum(theta*X_denom, axis=-1)
    numerator = ((p_numerator)*(1-2*d)**2 + d*(1-2*d)*(pA + p_denom) + d**2)*conditional + ((1 - 2*d)*p_numerator + d)*not_conditional
    denom = ((1 - 2*d)*p_denom  + d)*conditional + p_denom*not_conditional

    return jnp.divide(numerator, denom) # move division to after noise is added

def PTN_complex(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0]

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    X_A = jnp.stack([pA_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))
    conditional = jnp.array([is_cond(i) for i in trial])
    not_conditional = abs(1-conditional)

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter

    # Subject-level parameters/priors
    with numpyro.plate("subj", n_Ps):
        d_bases = numpyro.sample("d_base", dist.Normal(0, 1))
        d_deltas = numpyro.sample("d_delta", dist.Normal(0, 1)) # sum of these is approx uniform and each can span approx full [.05, .95]

    # Subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

    d_lin = d_bases[subj] + jnp.exp(d_deltas[subj]) * conjdisj # constrain d < d'
    d = sigmoid(d_lin)/2. # require this be below .50

    theta_ind = (subj*n_conds) + cond
    theta = thetas[theta_ind,:]

    p_ptn = prob_judge_PTN(trial, theta, X_num, X_denom, X_A, d) #

    # Likelihood
    with numpyro.plate("data", len(trial)):
#         yhat = numpyro.sample("yhat", dist.Normal(p_ptn, sigma), obs = y) # wrong but replicates paper
        yhat = numpyro.sample("yhat", dist.Beta(p_ptn*k, (1-p_ptn)*k), obs=y)

    return yhat


def PTN_simple(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0]

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    X_A = jnp.stack([pA_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))
    conditional = jnp.array([is_cond(i) for i in trial])
    not_conditional = abs(1-conditional)

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter

    # Subject-level parameters/priors
    with numpyro.plate("subj", n_Ps):
        ds = numpyro.sample("d_base", dist.Beta(1, 1))
#        d_deltas = numpyro.sample("d_delta", dist.Normal(0, 1)) # sum of these is approx uniform and each can span approx full [.05, .95]

    # Subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

    d_lin = ds[subj] # constrain d < d'
    d = d_lin/2. # require this be below .50

    theta_ind = (subj*n_conds) + cond
    theta = thetas[theta_ind,:]

    p_ptn = prob_judge_PTN(trial, theta, X_num, X_denom, X_A, d) #

    # Likelihood
    with numpyro.plate("data", len(trial)):
#         yhat = numpyro.sample("yhat", dist.Normal(p_ptn, sigma), obs = y) # wrong but replicates paper
        yhat = numpyro.sample("yhat", dist.Beta(p_ptn*k, (1-p_ptn)*k), obs=y)

    return yhat

## =====================================
## Hierarchical Models
## =====================================


## -------- Bayesian Sampler -----------

def bayesian_sampler_simple_mlm_d(data, y=None):

    # parameterized in terms of d and d' for comparison of model fit

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0]

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter

    d_base_pop = numpyro.sample("d_base_pop", dist.Normal(-1.0, 1.0))
 #   d_delta_pop = numpyro.sample("d_delta_pop", dist.Normal(0, .5)) # bias toward lower values for non conj/disj trials
    d_base_sd = numpyro.sample("d_base_sd", dist.LogNormal(-1., 1.)) # was halfcauchy(1)
#    d_delta_sd = numpyro.sample("d_delta_sd", dist.LogNormal(-1., 1.)) # approx uniform altogether we hope

    # subject-level parameters/priors
    with numpyro.plate("subj", n_Ps):
        d_bases = numpyro.sample("d_base_r", dist.Normal(0, 1)) # centered implementation

    # subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

    d_lin = d_base_pop + d_bases[subj]*d_base_sd  # exp() constrains d_delta to be positive
    d = sigmoid(d_lin)/3.0 # require this be in [0, 1/3]

    # numpyro.deterministic("N_subj", 1 + jnp.exp(N_prime_pop + N_primes))
    # numpyro.deterministic("N_prime_subj", 1 + jnp.exp(N_prime_pop + N_primes) + jnp.exp(N_delta_pop + N_deltas))

    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]

    p_bs = prob_judge_BS_d(theta, X_num, X_denom, d)

    # Likelihood
    with numpyro.plate("data", len(trial)):
        yhat = numpyro.sample("yhat", dist.Beta(p_bs*k, (1-p_bs)*k), obs=y)

    return yhat

def bayesian_sampler_complex_mlm_d(data, y=None):

    # parameterized in terms of d and d' for comparison of model fit

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0]

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter

    d_base_pop = numpyro.sample("d_base_pop", dist.Normal(-1.0, 1.0))
    d_delta_pop = numpyro.sample("d_delta_pop", dist.Normal(0, .5)) # bias toward lower values for non conj/disj trials
    d_base_sd = numpyro.sample("d_base_sd", dist.LogNormal(-1., 1.)) # was halfcauchy(1)
    d_delta_sd = numpyro.sample("d_delta_sd", dist.LogNormal(-1., 1.)) # approx uniform altogether we hope

    # subject-level parameters/priors
    with numpyro.plate("subj", n_Ps):
        d_bases = numpyro.sample("d_base_r", dist.Normal(0, 1))
        d_deltas = numpyro.sample("d_delta_r", dist.Normal(0, 1))

    # subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

    d_lin = (d_base_pop +
             d_bases[subj]*d_base_sd +
             jnp.exp(d_delta_pop + d_delta_sd*d_deltas[subj])*conjdisj
            )  # exp() constrains d_delta to be positive
    d = sigmoid(d_lin)/3.0 # require this be in [0, 1/3]

    numpyro.deterministic("d_subj", sigmoid(d_base_pop + d_bases*d_base_sd)/3.)
    numpyro.deterministic("d_prime_subj",
                          sigmoid(d_base_pop +
                                  d_bases*d_base_sd +
                                  jnp.exp(d_delta_pop + d_deltas*d_delta_sd)
                                 )/3.
                         )

    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]

    p_bs = prob_judge_BS_d(theta, X_num, X_denom, d)

    # Likelihood
    with numpyro.plate("data", len(trial)):
        yhat = numpyro.sample("yhat", dist.Beta(p_bs*k, (1-p_bs)*k), obs=y)

    return yhat


def bayesian_sampler_complex_mlm(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = jnp.unique(subj).shape[0], jnp.unique(cond).shape[0]

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
    beta_pop = numpyro.sample("beta_pop", dist.Normal(0, 1)) # roughly uniform after summing with random effects + sigmoid()
    beta_sd = numpyro.sample("beta_sd", dist.HalfCauchy(1))

    N_prime_pop = numpyro.sample("N_prime_pop", dist.Normal(0,2)) # mildly informative
    N_delta_pop = numpyro.sample("N_delta_pop", dist.Normal(0,2))
    N_prime_sd = numpyro.sample("N_prime_sd", dist.HalfCauchy(2))
    N_delta_sd = numpyro.sample("N_delta_sd", dist.HalfCauchy(2))

    # subject-level parameters/priors <--- maybe change to non-centered parameterization for all these
    with numpyro.plate("subj", n_Ps):
        betas = numpyro.sample("beta_r", dist.Normal(0, beta_sd))
        N_deltas = numpyro.sample("N_delta_r", dist.Normal(0, N_delta_sd))
        N_primes = numpyro.sample("N_prime_r", dist.Normal(0, N_prime_sd))

    # subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

    beta = sigmoid(beta_pop + betas[subj]) # constrains beta to [0,1]
    #beta = jnp.exp(beta_pop + betas[subj])
    numpyro.deterministic("beta_subj", jnp.exp(beta_pop + betas))

    # exp() needed to constrain N and N_delta positive
    N = 1 + jnp.exp(N_prime_pop + N_primes[subj]) + jnp.exp(N_delta_pop + N_deltas[subj]) * not_conjdisj # they also required N be at least 1

    numpyro.deterministic("N_subj", 1 + jnp.exp(N_prime_pop + N_primes))
    numpyro.deterministic("N_prime_subj", 1 + jnp.exp(N_prime_pop + N_primes) + jnp.exp(N_delta_pop + N_deltas))

    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]

    p_bs = prob_judge_BS(theta, X_num, X_denom, N, beta)

    # Likelihood
    with numpyro.plate("data", len(trial)):
        yhat = numpyro.sample("yhat", dist.Beta(p_bs*k, (1-p_bs)*k), obs=y)

    return yhat


## -------------- PT+N -----------------

def PTN_simple_mlm(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0]

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    X_A = jnp.stack([pA_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))
    conditional = jnp.array([is_cond(i) for i in trial])
    not_conditional = abs(1-conditional)

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
    d_base_pop = numpyro.sample("d_base_pop", dist.Normal(-1.0, 1.0))
#    d_delta_pop = numpyro.sample("d_delta_pop", dist.Normal(0, .5)) # bias toward lower values for non conj/disj trials
    d_base_sd = numpyro.sample("d_base_sd", dist.LogNormal(-1., 1.)) # was halfcauchy(1)
#    d_delta_sd = numpyro.sample("d_delta_sd", dist.LogNormal(-1., 1.)) # approx uniform altogether we hope

    # subject-level parameters/priors
    with numpyro.plate("subj", n_Ps):
        d_bases = numpyro.sample("d_base_r", dist.Normal(0, 1))

    # Subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

    d_lin = d_base_pop + d_bases[subj]*d_base_sd  # exp() constrains d_delta to be positive
    d = sigmoid(d_lin)/2 # require this be in [0,.50]

    theta_ind = (subj*n_conds) + cond
    theta = thetas[theta_ind,:]

    p_ptn = prob_judge_PTN(trial, theta, X_num, X_denom, X_A, d) #

    # deterministic sites

    numpyro.deterministic("delta_subj", sigmoid(d_base_pop + d_bases*d_base_sd)/2)

    # Likelihood
    with numpyro.plate("data", len(trial)):
#         yhat = numpyro.sample("yhat", dist.Normal(p_ptn, sigma), obs = y) # wrong but replicates paper
        yhat = numpyro.sample("yhat", dist.Beta(p_ptn*k, (1-p_ptn)*k), obs=y)

    return yhat


def PTN_complex_mlm(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0]

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    X_A = jnp.stack([pA_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))
    conditional = jnp.array([is_cond(i) for i in trial])
    not_conditional = abs(1-conditional)

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
    d_base_pop = numpyro.sample("d_base_pop", dist.Normal(-1.0, 1.0))
    d_delta_pop = numpyro.sample("d_delta_pop", dist.Normal(0, .5)) # bias toward lower values for non conj/disj trials
    d_base_sd = numpyro.sample("d_base_sd", dist.LogNormal(-1., 1.)) # was halfcauchy(1)
    d_delta_sd = numpyro.sample("d_delta_sd", dist.LogNormal(-1., 1.)) # approx uniform altogether we hope

    # subject-level parameters/priors
    with numpyro.plate("subj", n_Ps):
        d_bases = numpyro.sample("d_base_r", dist.Normal(0, 1))
        d_deltas = numpyro.sample("d_delta_r", dist.Normal(0, 1))

    # Subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

    d_lin = (d_base_pop +
             d_bases[subj]*d_base_sd +
             jnp.exp(d_delta_pop + d_delta_sd*d_deltas[subj])*conjdisj
            )  # exp() constrains d_delta to be positive
    d = sigmoid(d_lin)/2.0 # require this be in [0, .50]

    # deterministic sites
    numpyro.deterministic("d_subj", sigmoid(d_base_pop + d_bases*d_base_sd)/2.)
    numpyro.deterministic("d_prime_subj",
                          sigmoid(d_base_pop +
                                  d_bases*d_base_sd +
                                  jnp.exp(d_delta_pop + d_deltas*d_delta_sd)
                                 )/2.
                         )

    theta_ind = (subj*n_conds) + cond
    theta = thetas[theta_ind,:]

    p_ptn = prob_judge_PTN(trial, theta, X_num, X_denom, X_A, d) #


    # Likelihood
    with numpyro.plate("data", len(trial)):
#         yhat = numpyro.sample("yhat", dist.Normal(p_ptn, sigma), obs = y) # wrong but replicates paper
        yhat = numpyro.sample("yhat", dist.Beta(p_ptn*k, (1-p_ptn)*k), obs=y)

    return yhat

def calc_ptn_prob(prob, d):
    return ((1 - 2*d)*prob  + d)


def PTN_complex_mlm_simplecond(data, y=None):
    # an alternate version of PT+N that treats conditional probabilties like simple probabilities
    # equivalently, a Bayesian Sampler allowing beta parameter in [0, inf]

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0]

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    X_A = jnp.stack([pA_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))
    conditional = jnp.array([is_cond(i) for i in trial])
    not_conditional = abs(1-conditional)

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
    d_base_pop = numpyro.sample("d_base_pop", dist.Normal(-1.0, 1.0))
    d_delta_pop = numpyro.sample("d_delta_pop", dist.Normal(0, .5)) # bias toward lower values for non conj/disj trials
    d_base_sd = numpyro.sample("d_base_sd", dist.LogNormal(-1., 1.)) # was halfcauchy(1)
    d_delta_sd = numpyro.sample("d_delta_sd", dist.LogNormal(-1., 1.)) # approx uniform altogether we hope

    # subject-level parameters/priors
    with numpyro.plate("subj", n_Ps):
        d_bases = numpyro.sample("d_base_r", dist.Normal(0, 1))
        d_deltas = numpyro.sample("d_delta_r", dist.Normal(0, 1))

    # Subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

    d_lin = (d_base_pop +
             d_bases[subj]*d_base_sd +
             jnp.exp(d_delta_pop + d_delta_sd*d_deltas[subj])*conjdisj
            )  # exp() constrains d_delta to be positive
    d = sigmoid(d_lin)/2.0 # require this be in [0, .50]

    # deterministic sites
    numpyro.deterministic("d_subj", sigmoid(d_base_pop + d_bases*d_base_sd)/2.)
    numpyro.deterministic("d_prime_subj",
                          sigmoid(d_base_pop +
                                  d_bases*d_base_sd +
                                  jnp.exp(d_delta_pop + d_deltas*d_delta_sd)
                                 )/2.
                         )

    theta_ind = (subj*n_conds) + cond
    theta = thetas[theta_ind,:]

    p_ptn = calc_ptn_prob(calc_prob(theta, X_num, X_denom), d)

    # Likelihood
    with numpyro.plate("data", len(trial)):
#         yhat = numpyro.sample("yhat", dist.Normal(p_ptn, sigma), obs = y) # wrong but replicates paper
        yhat = numpyro.sample("yhat", dist.Beta(p_ptn*k, (1-p_ptn)*k), obs=y)

    return yhat



## =====================================
## Trial-level categorical mdoels (rounding to nearest 5)
## =====================================


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
    
    a = mu*N + 1.  # updated 9/6/22, 4:20 PM
    b = (1.-mu)*N + 1.
    
    n_resps = (responses.shape[0]-1)
    step = int(20/n_resps)
    rnd_unit_scaled = 1/n_resps
    
    lower = jnp.clip((responses/n_resps) - rnd_unit_scaled/2., 1e-8, 1-1e-8)
    upper = jnp.clip((responses/n_resps) + rnd_unit_scaled/2., 1e-8, 1-1e-8)
    
    prob_resps = bs_dist_cdf(N, beta, a, b, upper) - bs_dist_cdf(N, beta, a, b, lower)
    prob_resps = (spread_vec(prob_resps, step) + 1e-30)
    prob_resps = (prob_resps)/jnp.sum(prob_resps)
    
    return(prob_resps)

def f(mu, k, responses):
    
    a = mu*k + 1. # 8/26/22, 5:38 PM new PTN version
    b = (1.-mu)*k + 1.
    
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


lbeta_cat_probs = vmap(f, (0, 0, None)) # change to map for k

bs_cat_probs = vmap(f_bs, (0, 0, 0, None))

responses_10 = jnp.linspace(0, 10, num=11)
responses_5 = jnp.linspace(0, 20, num=21)

# def bs_complex_mlm_trial_level(data, y=None):

#     # Data processing
#     trial, subj, cond = data["trial"], data["subj"], data["cond"]
#     n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0] 

#     # setup "design matrix" (of sorts)
#     X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
#     conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))

#     # population level parameters/priors
#     # k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
#     beta_pop = numpyro.sample("beta_pop", dist.Normal(-1, 1)) # skewed after sigmoid
#     beta_sd = numpyro.sample("beta_sd", dist.LogNormal(-1, 1))

#     N_prime_pop = numpyro.sample("N_prime_pop", dist.Normal(10, 10)) # mildly informative
#     N_delta_pop = numpyro.sample("N_delta_pop", dist.Normal(0, 20)) 
#     N_prime_sd = numpyro.sample("N_prime_sd", dist.LogNormal(-.5, 1.5))
#     N_delta_sd = numpyro.sample("N_delta_sd", dist.LogNormal(-.5, 1.5))
    
#     rnd_policy = numpyro.sample("rnd_policy", dist.Dirichlet(jnp.ones(3)))

#     # subject-level parameters/priors <--- non-centered parameterization for all these
#     with numpyro.plate("subj", n_Ps):
#         betas = numpyro.sample("beta_r", dist.Normal(0, 1))*beta_sd 
#         N_deltas = numpyro.sample("N_delta_r", dist.Normal(0, 1))*N_delta_sd
#         N_primes = numpyro.sample("N_prime_r", dist.Normal(0, 1))*N_prime_sd

#     # subject/query-level parameters/priors
#     with numpyro.plate("cond", n_Ps*n_conds):
#         thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

#     # beta = expit(beta_pop + betas[subj])*10 # constrains beta to [0,10]
    
#     beta = softplus(beta_pop + betas[subj])
    
#     numpyro.deterministic("beta_subj", softplus(beta_pop + betas))

#     # exp() needed to constrain N and N_delta positive
#     N = 1 + softplus(N_prime_pop + N_primes[subj]) + softplus(N_delta_pop + N_deltas[subj]) * not_conjdisj # they also required N be at least 1

#     numpyro.deterministic("N_prime_subj", 1 + softplus(N_prime_pop + N_primes))
#     numpyro.deterministic("N_subj", 1 + softplus(N_prime_pop + N_primes) + softplus(N_delta_pop + N_deltas))
#     # numpyro.deterministic("beta_subj", expit(beta_pop + betas)*10)

#     theta_ind = ((subj*n_conds)+cond)
#     theta = thetas[theta_ind,:]
    
#     pi = calc_prob(theta, X_num, X_denom)

#     # Likelihood
#     with numpyro.plate("data", len(trial)):
        
#         resp_probs = (
#         1./21.*rnd_policy[0] +
#         bs_cat_probs(pi, N, beta, responses_5)*rnd_policy[1] + 
#         bs_cat_probs(pi, N, beta, responses_10)*rnd_policy[2]
#         )
        
#         yhat = numpyro.sample("yhat", dist.Categorical(probs=resp_probs), obs=y) # rounded


#     return yhat


def bs_complex_mlm_trial_level(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0] 

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
    # k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
    beta_pop = numpyro.sample("beta_pop", dist.Normal(-.5, .4))  # dist.Normal(-1, 1)) for softplus
    beta_sd = numpyro.sample("beta_sd", dist.LogNormal(-1, .5)) # dist.LogNormal(-1, 1)) for softplus

    N_prime_pop = numpyro.sample("N_prime_pop", dist.Normal(1, 1)) # dist.Normal(10, 5)) for softplus
    N_delta_pop = numpyro.sample("N_delta_pop", dist.Normal(0, 1))  # dist.Normal(0, 10)) for softplus
    N_prime_sd = numpyro.sample("N_prime_sd", dist.LogNormal(-.5, .5)) # dist.LogNormal(-.5, 1.6)) for softplus
    N_delta_sd = numpyro.sample("N_delta_sd", dist.LogNormal(-.5, .5)) # dist.LogNormal(-.5, 1.6)) for softplus
    
    rnd_policy = numpyro.sample("rnd_policy", dist.Dirichlet(jnp.ones(3)))

    # subject-level parameters/priors <--- non-centered parameterization for all these
    with numpyro.plate("subj", n_Ps):
        betas = numpyro.sample("beta_r", dist.Normal(0, 1))*beta_sd 
        N_deltas = numpyro.sample("N_delta_r", dist.Normal(0, 1))*N_delta_sd
        N_primes = numpyro.sample("N_prime_r", dist.Normal(0, 1))*N_prime_sd

    # subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

    # beta = expit(beta_pop + betas[subj])*10 # constrains beta to [0,10]
    
    beta = jnp.exp(beta_pop + betas[subj]) # from softplus to exp
    
    numpyro.deterministic("beta_subj", jnp.exp(beta_pop + betas))

    # exp() needed to constrain N and N_delta positive
    N = 1 + jnp.exp(N_prime_pop + N_primes[subj]) + jnp.exp(N_delta_pop + N_deltas[subj]) * not_conjdisj # they also required N be at least 1
    

    numpyro.deterministic("N_prime_subj", 1 + jnp.exp(N_prime_pop + N_primes))
    numpyro.deterministic("N_subj", 1 + jnp.exp(N_prime_pop + N_primes) + jnp.exp(N_delta_pop + N_deltas))
    
    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]
    
    pi = calc_prob(theta, X_num, X_denom)

    # Likelihood
    with numpyro.plate("data", len(trial)):
        
        resp_probs = (
        1./21.*rnd_policy[0] +
        bs_cat_probs(pi, N, beta, responses_5)*rnd_policy[1] + 
        bs_cat_probs(pi, N, beta, responses_10)*rnd_policy[2]
        )
        
        yhat = numpyro.sample("yhat", dist.Categorical(probs=resp_probs), obs=y) # rounded


    return yhat


def bs_complex_mlm_trial_level_varyN(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0] 

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
    # k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
    beta_pop = numpyro.sample("beta_pop", dist.Normal(-.5, .4))  # dist.Normal(-1, 1)) for softplus
    beta_sd = numpyro.sample("beta_sd", dist.LogNormal(-1, .5)) # dist.LogNormal(-1, 1)) for softplus

    N_prime_pop = numpyro.sample("N_prime_pop", dist.Normal(1, 1)) # dist.Normal(10, 5)) for softplus
    N_delta_pop = numpyro.sample("N_delta_pop", dist.Normal(0, 1))  # dist.Normal(0, 10)) for softplus
    N_prime_sd = numpyro.sample("N_prime_sd", dist.LogNormal(-.5, .5)) # dist.LogNormal(-.5, 1.6)) for softplus
    N_delta_sd = numpyro.sample("N_delta_sd", dist.LogNormal(-.5, .5)) # dist.LogNormal(-.5, 1.6)) for softplus
    
    N_trial_sd = numpyro.sample("N_trial_sd", dist.LogNormal(-1, .3))
    
    rnd_policy = numpyro.sample("rnd_policy", dist.Dirichlet(jnp.ones(3)))

    # subject-level parameters/priors <--- non-centered parameterization for all these
    with numpyro.plate("subj", n_Ps):
        betas = numpyro.sample("beta_r", dist.Normal(0, 1))*beta_sd 
        N_deltas = numpyro.sample("N_delta_r", dist.Normal(0, 1))*N_delta_sd
        N_primes = numpyro.sample("N_prime_r", dist.Normal(0, 1))*N_prime_sd

    # subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

    # beta = expit(beta_pop + betas[subj])*10 # constrains beta to [0,10]
    
    beta = jnp.exp(beta_pop + betas[subj]) # from softplus to exp
    
    numpyro.deterministic("beta_subj", jnp.exp(beta_pop + betas))

    # exp() needed to constrain N and N_delta positive
#     N = 1 + jnp.exp(N_prime_pop + N_primes[subj]) + jnp.exp(N_delta_pop + N_deltas[subj]) * not_conjdisj # they also required N be at least 1
    

    numpyro.deterministic("N_prime_subj", 1 + jnp.exp(N_prime_pop + N_primes))
    numpyro.deterministic("N_subj", 1 + jnp.exp(N_prime_pop + N_primes) + jnp.exp(N_delta_pop + N_deltas))
    
    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]
    
    pi = calc_prob(theta, X_num, X_denom)

    # Likelihood
    with numpyro.plate("data", len(trial)):
        N_trial = numpyro.sample("N_trial", dist.Normal(0, 1))*N_trial_sd
        
        N = 1 + jnp.exp(N_prime_pop + N_primes[subj]) + jnp.exp(N_delta_pop + N_deltas[subj]) * not_conjdisj # they also required N be at least 1
        N = N * jnp.exp(N_trial)
        
        resp_probs = (
        1./21.*rnd_policy[0] +
        bs_cat_probs(pi, N, beta, responses_5)*rnd_policy[1] + 
        bs_cat_probs(pi, N, beta, responses_10)*rnd_policy[2]
        )
        
        yhat = numpyro.sample("yhat", dist.Categorical(probs=resp_probs), obs=y) # rounded


    return yhat


def ptn_simplecond_mlm_trial_level_disc(data, y=None):
    
    # parameterized in terms of d and d' for comparison of model fit

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0] 
    
    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
#     k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
    rnd_policy = numpyro.sample("rnd_policy", dist.Dirichlet(jnp.ones(3)))
    
    d_base_pop = numpyro.sample("d_base_pop", dist.Normal(-1.0, 1.0))
    d_delta_pop = numpyro.sample("d_delta_pop", dist.Normal(0, .5)) # bias toward lower values for non conj/disj trials
    d_base_sd = numpyro.sample("d_base_sd", dist.LogNormal(-1., 1.)) # was halfcauchy(1)
    d_delta_sd = numpyro.sample("d_delta_sd", dist.LogNormal(-1., 1.)) # approx uniform altogether we hope

    # subject-level parameters/priors 
    with numpyro.plate("subj", n_Ps):
        d_bases = numpyro.sample("d_base_r", dist.Normal(0, 1))
        d_deltas = numpyro.sample("d_delta_r", dist.Normal(0, 1))
        ks = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
        
    # subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))
    
    d_lin = (d_base_pop + 
             d_bases[subj]*d_base_sd + 
             jnp.exp(d_delta_pop + d_delta_sd*d_deltas[subj])*conjdisj
            )  # exp() constrains d_delta to be positive
    
    d = sigmoid(d_lin)/2.0
    
    numpyro.deterministic("d_subj", sigmoid(d_base_pop + d_bases*d_base_sd)/2.)
    numpyro.deterministic("d_prime_subj", 
                          sigmoid(d_base_pop + 
                                  d_bases*d_base_sd + 
                                  jnp.exp(d_delta_pop + d_deltas*d_delta_sd)
                                 )/2.
                         )
    
    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]
        
    p_bs = prob_judge_BS_d(theta, X_num, X_denom, d)
    k = ks[subj] # fixed/updated 8/26/22, 6:17 PM
    
    resp_probs = (
        1./21.*rnd_policy[0] +
        lbeta_cat_probs(p_bs, k, responses_5)*rnd_policy[1] + 
        lbeta_cat_probs(p_bs, k, responses_10)*rnd_policy[2]
    )

    # Likelihood
    with numpyro.plate("data", len(trial)):
        yhat = numpyro.sample("yhat", dist.Categorical(probs=resp_probs), obs=y) # rounded
        return(yhat)
    
    
def ptn_simplecond_mlm_trial_level_disc2(data, y=None):
    
    # parameterized in terms of d and d' for comparison of model fit

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0] 
    
    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
#     k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
    rnd_policy = numpyro.sample("rnd_policy", dist.Dirichlet(jnp.ones(5)))
    
    d_base_pop = numpyro.sample("d_base_pop", dist.Normal(-1.0, 1.0))
    d_delta_pop = numpyro.sample("d_delta_pop", dist.Normal(0, .5)) # bias toward lower values for non conj/disj trials
    d_base_sd = numpyro.sample("d_base_sd", dist.LogNormal(-1., 1.)) # was halfcauchy(1)
    d_delta_sd = numpyro.sample("d_delta_sd", dist.LogNormal(-1., 1.)) # approx uniform altogether we hope

    # subject-level parameters/priors 
    with numpyro.plate("subj", n_Ps):
        d_bases = numpyro.sample("d_base_r", dist.Normal(0, 1))
        d_deltas = numpyro.sample("d_delta_r", dist.Normal(0, 1))
        ks = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
        
    # subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))
    
    d_lin = (d_base_pop + 
             d_bases[subj]*d_base_sd + 
             jnp.exp(d_delta_pop + d_delta_sd*d_deltas[subj])*conjdisj
            )  # exp() constrains d_delta to be positive
    
    d = sigmoid(d_lin)/2.0
    
    numpyro.deterministic("d_subj", sigmoid(d_base_pop + d_bases*d_base_sd)/2.)
    numpyro.deterministic("d_prime_subj", 
                          sigmoid(d_base_pop + 
                                  d_bases*d_base_sd + 
                                  jnp.exp(d_delta_pop + d_deltas*d_delta_sd)
                                 )/2.
                         )
    
    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]
        
    p_bs = prob_judge_BS_d(theta, X_num, X_denom, d)
    k = ks[subj] # fixed/updated 8/26/22, 6:17 PM
    
    resp_probs = (
        1./21.*rnd_policy[0] +
        lbeta_cat_probs(p_bs, k, responses_5)*rnd_policy[1] + 
        lbeta_cat_probs(p_bs, k, responses_10)*rnd_policy[2] +
        lbeta_cat_probs(p_bs, k, responses_25)*rnd_policy[3] +
        lbeta_cat_probs(p_bs, k, responses_50)*rnd_policy[4]
    )

    # Likelihood
    with numpyro.plate("data", len(trial)):
        yhat = numpyro.sample("yhat", dist.Categorical(probs=resp_probs), obs=y) # rounded
        return(yhat)

## =====================================
## Hierarchical Mixture Models
## =====================================

import numpyro.distributions.constraints as constraints
from numpyro.distributions.util import promote_shapes, is_prng_key, validate_sample
from jax import lax
import jax

class MixtureBeta(dist.Distribution):
    def __init__(self, concentration1, concentration0, mixing_probs, validate_args=None):
        expand_shape = jax.lax.broadcast_shapes(
            jnp.shape(concentration1), jnp.shape(concentration0), jnp.shape(mixing_probs))
        self._beta = dist.Beta(concentration1, concentration0).expand(expand_shape)
        self._categorical = dist.Categorical(jnp.broadcast_to(mixing_probs, expand_shape))
        super(MixtureBeta, self).__init__(batch_shape=expand_shape[:-1], validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        key, key_idx = random.split(key)
        samples = self._beta.sample(key, sample_shape)
        ind = self._categorical.sample(key_idx, sample_shape)
        return jnp.take_along_axis(samples, ind[..., None], -1)[..., 0]

    def log_prob(self, value):
        dirichlet_probs = self._beta.log_prob(value[..., None])
        sum_probs = self._categorical.logits + dirichlet_probs
        return jax.nn.logsumexp(sum_probs, axis=-1)



## -------- Bayesian Sampler -----------

def bayesian_sampler_complex_mlm_mix(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds, n_obs = np.unique(subj).shape[0], np.unique(cond).shape[0], subj.shape[0]

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    not_conjdisj = abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfNormal(50)) # noise parameter
    beta_pop = numpyro.sample("beta_pop", dist.Normal(0, 1)) # roughly uniform after summing with random effects + sigmoid()
    beta_sd = numpyro.sample("beta_sd", dist.HalfCauchy(1))

    N_prime_pop = numpyro.sample("N_prime_pop", dist.Normal(0,2)) # mildly informative
    N_delta_pop = numpyro.sample("N_delta_pop", dist.Normal(0,2))
    N_prime_sd = numpyro.sample("N_prime_sd", dist.HalfCauchy(1))
    N_delta_sd = numpyro.sample("N_delta_sd", dist.HalfCauchy(1))

    # population mixture parameter
    mixing_probs = numpyro.sample("mixing", dist.Dirichlet(jnp.ones(2)))
    # mixing_probs = jnp.array([.95,.05])


    # k_50 = numpyro.sample("k_50", dist.TruncatedNormal(low=0., loc=600, scale = 100))
    # mix_kval = numpyro.sample("k_50", dist.HalfCauchy(300))

    k_50 = numpyro.sample("k_50", dist.Normal(0,20))
    mix_kval = 600+k_50
    mix_muval = .50

    # subject-level parameters/priors
    with numpyro.plate("subj", n_Ps):
        betas = numpyro.sample("beta_r", dist.Normal(0, beta_sd))
        N_deltas = numpyro.sample("N_delta_r", dist.Normal(0, N_delta_sd))
        N_primes = numpyro.sample("N_prime_r", dist.Normal(0, N_prime_sd))

    # subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

#     beta = sigmoid(beta_pop + betas[subj]) # constrains beta to [0,1]
    beta = jnp.exp(beta_pop + betas[subj])

    # exp() needed to constrain N and N_delta positive
    N = 1 + jnp.exp(N_prime_pop + N_primes[subj]) + jnp.exp(N_delta_pop + N_deltas[subj]) * not_conjdisj # they also required N be at least 1

    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]

    p_bs = prob_judge_BS(theta, X_num, X_denom, N, beta)

    # Likelihood
    with numpyro.plate("data", len(trial)):
        yhat = jnp.stack([p_bs, jnp.ones(n_obs)*mix_muval], -1)
        mix_k = jnp.stack([jnp.ones(n_obs)*k, jnp.ones(n_obs)*mix_kval], -1)
        numpyro.sample("yhat", MixtureBeta(yhat*mix_k, (1-yhat)*mix_k, mixing_probs), obs=y)

    return yhat

## -------------- PT+N -----------------

def PTN_complex_mlm_mix(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds, n_obs = np.unique(subj).shape[0], np.unique(cond).shape[0], subj.shape[0]

    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    X_A = jnp.stack([pA_vecs[i] for i in trial])
    conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))
    conditional = jnp.array([is_cond(i) for i in trial])
    not_conditional = abs(1-conditional)

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter

    d_base_pop = numpyro.sample("d_base_pop", dist.Normal(-1.1, 1.6))
    d_delta_pop = numpyro.sample("d_delta_pop", dist.Normal(0, .5)) # together approx uniform
    d_base_sd = numpyro.sample("d_base_sd", dist.HalfCauchy(1))
    d_delta_sd = numpyro.sample("d_delta_sd", dist.HalfCauchy(1))

    # population mixture parameters
    mixing_probs = numpyro.sample("mixing", dist.Dirichlet(jnp.ones(2)))
    k_50 = numpyro.sample("k_50", dist.Normal(0,20))
    mix_kval = 600+k_50
    mix_muval = .50

    # subject-level parameters/priors
    with numpyro.plate("subj", n_Ps):
        d_bases = numpyro.sample("d_base_r", dist.Normal(0, d_base_sd))
        d_deltas = numpyro.sample("d_delta_r", dist.Normal(0, d_delta_sd))

    # Subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))

    d_lin = d_base_pop + d_bases[subj] + jnp.exp(d_delta_pop + d_deltas[subj])*conjdisj  # exp() constrains d_delta to be positive
    d = sigmoid(d_lin)/2 # require this be in [0,.50]

    theta_ind = (subj*n_conds) + cond
    theta = thetas[theta_ind,:]

    p_ptn = prob_judge_PTN(trial, theta, X_num, X_denom, X_A, d) #

    # Likelihood
    with numpyro.plate("data", len(trial)):
        yhat = jnp.stack([p_ptn, jnp.ones(n_obs)*mix_muval], -1)
        mix_k = jnp.stack([jnp.ones(n_obs)*k, jnp.ones(n_obs)*mix_kval], -1)
        numpyro.sample("yhat", MixtureBeta(yhat*mix_k, (1-yhat)*mix_k, mixing_probs), obs=y)

    return yhat


#### --- unused

# lbeta_cat_probs = vmap(f, (0, 0, None)) # change to map for k

# # bs_cat_probs = vmap(f_bs, (0, 0, 0, None))

# responses_50 = jnp.linspace(0, 3, num=4)
# responses_25 = jnp.linspace(0, 5, num=6)
# responses_10 = jnp.linspace(0, 10, num=11)
# responses_5 = jnp.linspace(0, 20, num=21)

### a version with more rounding options
# def ptn_simplecond_mlm_trial_level_disc2(data, y=None):
    
#     # parameterized in terms of d and d' for comparison of model fit

#     # Data processing
#     trial, subj, cond = data["trial"], data["subj"], data["cond"]
#     n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0] 
    
#     # setup "design matrix" (of sorts)
#     X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
#     conjdisj, not_conjdisj = jnp.array([is_conjdisj(i) for i in trial]), abs(1-jnp.array([is_conjdisj(i) for i in trial]))

#     # population level parameters/priors
# #     k = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
#     rnd_policy = numpyro.sample("rnd_policy", dist.Dirichlet(jnp.ones(5)))
    
#     d_base_pop = numpyro.sample("d_base_pop", dist.Normal(-1.0, 1.0))
#     d_delta_pop = numpyro.sample("d_delta_pop", dist.Normal(0, .5)) # bias toward lower values for non conj/disj trials
#     d_base_sd = numpyro.sample("d_base_sd", dist.LogNormal(-1., 1.)) # was halfcauchy(1)
#     d_delta_sd = numpyro.sample("d_delta_sd", dist.LogNormal(-1., 1.)) # approx uniform altogether we hope

#     # subject-level parameters/priors 
#     with numpyro.plate("subj", n_Ps):
#         d_bases = numpyro.sample("d_base_r", dist.Normal(0, 1))
#         d_deltas = numpyro.sample("d_delta_r", dist.Normal(0, 1))
#         ks = numpyro.sample("k", dist.HalfCauchy(20)) # noise parameter
        
#     # subject/query-level parameters/priors
#     with numpyro.plate("cond", n_Ps*n_conds):
#         thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))
    
#     d_lin = (d_base_pop + 
#              d_bases[subj]*d_base_sd + 
#              jnp.exp(d_delta_pop + d_delta_sd*d_deltas[subj])*conjdisj
#             )  # exp() constrains d_delta to be positive
    
#     d = sigmoid(d_lin)/2.0
    
#     numpyro.deterministic("d_subj", sigmoid(d_base_pop + d_bases*d_base_sd)/2.)
#     numpyro.deterministic("d_prime_subj", 
#                           sigmoid(d_base_pop + 
#                                   d_bases*d_base_sd + 
#                                   jnp.exp(d_delta_pop + d_deltas*d_delta_sd)
#                                  )/2.
#                          )
    
#     theta_ind = ((subj*n_conds)+cond)
#     theta = thetas[theta_ind,:]
        
#     p_bs = prob_judge_BS_d(theta, X_num, X_denom, d)
#     k = ks[subj] # fixed/updated 8/26/22, 6:17 PM
    
#     resp_probs = (
#         1./21.*rnd_policy[0] +
#         lbeta_cat_probs(p_bs, k, responses_5)*rnd_policy[1] + 
#         lbeta_cat_probs(p_bs, k, responses_10)*rnd_policy[2] +
#         lbeta_cat_probs(p_bs, k, responses_25)*rnd_policy[3] +
#         lbeta_cat_probs(p_bs, k, responses_50)*rnd_policy[4]
#     )

#     # Likelihood
#     with numpyro.plate("data", len(trial)):
#         yhat = numpyro.sample("yhat", dist.Categorical(probs=resp_probs), obs=y) # rounded
#         return(yhat)