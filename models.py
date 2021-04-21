## -------------------------------------
## Relative Frequency
## -------------------------------------

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
        
    numerator = jnp.sum(theta*X_num, axis=-1)
    denom = jnp.sum(theta*X_denom, axis=-1)
    pi = jnp.divide(numerator, denom)

    with numpyro.plate("data", len(trial)):
#         yhat = numpyro.sample("yhat", dist.Normal(pi, sigma), obs = y) # wrong but replicates paper
        yhat = numpyro.sample("yhat", dist.Beta(pi*k, (1-pi)*k), obs=y)
    
    return yhat

## -------------------------------------
## Bayesian Sampler
## -------------------------------------

def calc_prob(theta, X_num, X_denom):

    numerator = jnp.sum(theta*X_num, axis=-1)
    denom = jnp.sum(theta*X_denom, axis=-1)
    
    return jnp.divide(numerator, denom)


def prob_judge_BS(theta, X_num, X_denom, N, beta):

    pi = calc_prob(theta, X_num, X_denom)
    p_bs = pi * N / (N + 2.*beta) + beta/(N + 2.*beta)
    
    return p_bs



def bayesian_sampler_complex(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0] 
    
    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    not_conjdisj = abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfNormal(50)) # noise parameter

    # subject-level parameters/priors 
    with numpyro.plate("subj", n_Ps):
#         betas = numpyro.sample("beta", dist.Uniform(0.0, 1.0)) # bounded to [0,1] as in paper
        betas = numpyro.sample("beta", dist.LogNormal(-1., 1.))
#         N_deltas = numpyro.sample("N_delta", dist.HalfNormal(50))
#         N_primes = numpyro.sample("N_prime", dist.HalfNormal(50))
        N_deltas = numpyro.sample("N_delta", dist.Normal(0,3)) # 3/25/21, 11:10 PM changed this
        N_primes = numpyro.sample("N_prime", dist.Normal(0,3))
        
    # subject/query-level parameters/priors
    with numpyro.plate("cond", n_Ps*n_conds):
        thetas = numpyro.sample("theta", dist.Dirichlet(jnp.ones(4)))
    
    beta = betas[subj] # use jnp.exp() if unbounded
    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]
#     N = 1 + N_primes[subj] + N_deltas[subj] * not_conjdisj 
    N_lin = N_primes[subj] + N_deltas[subj] * not_conjdisj #  < -------------- gah, fix this 3/26/21, 9:41 AM
    N = 1 + jnp.exp(N_lin) # they required N be at least 1
    
    p_bs = prob_judge_BS(theta, X_num, X_denom, N, beta)
    
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


## =====================================
## Hierarchical Models
## =====================================


## -------- Bayesian Sampler -----------

def bayesian_sampler_complex_mlm(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0] 
    
    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    not_conjdisj = abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfNormal(50)) # noise parameter
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
    
#     beta = sigmoid(beta_pop + betas[subj]) # constrains beta to [0,1]
    beta = jnp.exp(beta_pop + betas[subj])
    
    # exp() needed to constrain N and N_delta positive
    N = 1 + jnp.exp(N_prime_pop + N_primes[subj]) + jnp.exp(N_delta_pop + N_deltas[subj]) * not_conjdisj # they also required N be at least 1
    
    theta_ind = ((subj*n_conds)+cond)
    theta = thetas[theta_ind,:]
        
    p_bs = prob_judge_BS(theta, X_num, X_denom, N, beta)
    
    # record deterministic sites for subj-level values
    numpyro.deterministic("beta_subj", jnp.exp(beta_pop + betas))
    numpyro.deterministic("N_subj", 1 + jnp.exp(N_prime_pop + N_primes))
    numpyro.deterministic("N_prime_subj", 1 + jnp.exp(N_prime_pop + N_primes) + jnp.exp(N_delta_pop + N_deltas))
    
    # Likelihood
    with numpyro.plate("data", len(trial)):
        yhat = numpyro.sample("yhat", dist.Beta(p_bs*k, (1-p_bs)*k), obs=y)
    
    return yhat



def bayesian_sampler_complex_mlm_restricted(data, y=None):

    # Data processing
    trial, subj, cond = data["trial"], data["subj"], data["cond"]
    n_Ps, n_conds = np.unique(subj).shape[0], np.unique(cond).shape[0] 
    
    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    not_conjdisj = abs(1-jnp.array([is_conjdisj(i) for i in trial]))

    # population level parameters/priors
    k = numpyro.sample("k", dist.HalfNormal(50)) # noise parameter
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
    
    beta = sigmoid(beta_pop + betas[subj]) # constrains beta to [0,1]/
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
    k = numpyro.sample("k", dist.HalfNormal(50)) # noise parameter
    d_base_pop = numpyro.sample("d_base_pop", dist.Normal(-1.0, 1.0))
    d_delta_pop = numpyro.sample("d_delta_pop", dist.Normal(0, .5)) # bias toward lower values for non conj/disj trials
    d_base_sd = numpyro.sample("d_base_sd", dist.LogNormal(-1., 1.)) # was halfcauchy(1)
    d_delta_sd = numpyro.sample("d_delta_sd", dist.LogNormal(-1., 1.)) # approx uniform altogether we hope

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

    # deterministic sites
    
    numpyro.deterministic("delta_subj", sigmoid(d_base_pop + d_bases)/2)
    numpyro.deterministic("delta_prime_subj", sigmoid(d_base_pop + d_bases + jnp.exp(d_delta_pop + d_deltas))/2 )
    # Likelihood
    with numpyro.plate("data", len(trial)):
#         yhat = numpyro.sample("yhat", dist.Normal(p_ptn, sigma), obs = y) # wrong but replicates paper
        yhat = numpyro.sample("yhat", dist.Beta(p_ptn*k, (1-p_ptn)*k), obs=y)
    
    return yhat

def calc_ptn_prob(prob, d):
    return ((1 - 2*d)*prob  + d)


def PTN_complex_mlm_simplecond(data, y=None):
    # an alternate version of PT+N that treats conditional probabilties like simple probabilities

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
    k = numpyro.sample("k", dist.HalfNormal(50)) # noise parameter
    d_base_pop = numpyro.sample("d_base_pop", dist.Normal(-1.0, 1.0))
    d_delta_pop = numpyro.sample("d_delta_pop", dist.Normal(0, .5)) # bias toward lower values for non conj/disj trials
    d_base_sd = numpyro.sample("d_base_sd", dist.LogNormal(-1., 1.)) # was halfcauchy(1)
    d_delta_sd = numpyro.sample("d_delta_sd", dist.LogNormal(-1., 1.)) # approx uniform altogether we hope

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
    
#    p_ptn = prob_judge_PTN(trial, theta, X_num, X_denom, X_A, d) #
    p_ptn = calc_ptn_prob(calc_prob(theta, X_num, X_denom), d)

    # Likelihood
    with numpyro.plate("data", len(trial)):
#         yhat = numpyro.sample("yhat", dist.Normal(p_ptn, sigma), obs = y) # wrong but replicates paper
        yhat = numpyro.sample("yhat", dist.Beta(p_ptn*k, (1-p_ptn)*k), obs=y)
    
    return yhat

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
    k = numpyro.sample("k", dist.HalfNormal(50)) # noise parameter

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
