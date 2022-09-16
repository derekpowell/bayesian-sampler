import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist

import siuba as s
from siuba import _


from plotnine import * 

from lib.models import calc_prob, prob_judge_PTN
from lib.helpers import *

def p_to_bs(p, beta, N):
    return (p * N + beta)/(N + 2.*beta)


def p_to_ptn(p, d):
    # code here
    return((1-2*d)*p + d)


def sim_bayes_samp_trial(prob, beta, N):
    # p = dist.Beta(prob*N, (1-prob)*N).sample(PRNGKey(0), )
    p = np.random.beta(1+prob*N, 1+(1-prob)*N, size=len(prob)) # updated 9/13/22, 3:48 PM
    res = (p * N + beta)/(N + 2.*beta)
    return np.asarray(res)


def sim_ptn_samp_trial(prob, d, N):
    # will round N
#     S = np.random.binomial(np.round(N).astype("int"), (1-2*d)*prob + d)
#     res = S/np.round(N).astype("int")
    # updated 9/13/22, 3:49 PM
    prob = (1-2*d)*prob + d
    res = np.random.beta(1+prob*N, 1+(1-prob)*N, size=len(prob))
    return np.asarray(res)


trial_types = [
    'A',
    'AandB',
    'AandnotB',
    'AgB',
    'AgnotB',
    'AorB',
    'AornotB',
    'B',
    'BgA',
    'BgnotA',
    'notA',
    'notAandB',
    'notAandnotB',
    'notAgB',
    'notAgnotB',
    'notAorB',
    'notAornotB',
    'notB',
    'notBgA',
    'notBgnotA'
    ]

## ------
## simulate data into pandas DataFrame
## -------

def make_sim_data_bs_trials(n_participants, n_blocks, n_conditions, params):
    all_thetas = make_thetas(n_participants)

    trial = list()
    cond = list()
    block = list()
    subj  = list()

    for p in range(0, n_participants):
        for b in range(0, n_blocks):
            for c in range(0, n_conditions):
                for t in trial_types:
                    trial.append(t)
                    cond.append(c)
                    block.append(b)
                    subj.append(p)
                    
    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])

    sim_data = pd.DataFrame(
            data = {
                "ID": subj,
                "querytype": trial, 
                "block": block,
                "condition": cond
            }) >> s.mutate(
                conjdisj_trial = _.querytype.apply(is_conjdisj),
                N_prime = _.ID.apply(lambda x: params["N_prime"][x]),
                N_delta = _.ID.apply(lambda x: params["N_delta"][x]),
                beta = _.ID.apply(lambda x: params["beta"][x]),
                theta = _.ID.apply(lambda x: all_thetas[x])
            ) >> s.mutate(N = _.N_prime + _.N_delta * abs(1-_.conjdisj_trial))

    sim_data["prob"] = calc_prob(np.asarray(all_thetas)[np.asarray(subj)], X_num, X_denom)
    sim_data["estimate"] = sim_bayes_samp_trial(np.asarray(sim_data.prob), np.asarray(sim_data.beta),  np.asarray(sim_data.N)).astype("float")
    
    return sim_data




def make_sim_data_bs_avg(n_participants, n_conditions, params, k = 20.):
    ## simulate average trial-level responses with a fixed amount of noise (untested 8/9/22, 1:24 PM)
    ## simulates bayesian sampler predictions (or simple conditional PTN judgments)
    
    all_thetas = make_thetas(n_participants)

    trial = list()
    cond = list()

    subj  = list()

    for p in range(0, n_participants):
            for c in range(0, n_conditions):
                for t in trial_types:
                    trial.append(t)
                    cond.append(c)

                    subj.append(p)
                    
    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])

    sim_data = pd.DataFrame(
            data = {
                "ID": subj,
                "querytype": trial, 
                "condition": cond
            }) >> s.mutate(
                conjdisj_trial = _.querytype.apply(is_conjdisj),
                N_base = _.ID.apply(lambda x: params["N_base"][x]),
                N_delta = _.ID.apply(lambda x: params["N_delta"][x]),
                beta = _.ID.apply(lambda x: params["beta"][x]),
                theta = _.ID.apply(lambda x: all_thetas[x])
            ) >> s.mutate(
        N = _.N_base + _.N_delta * abs(1-_.conjdisj_trial),
        implied_d = _.beta / (2*_.beta + _.N)
    )

    sim_data["prob"] = calc_prob(np.asarray(all_thetas)[np.asarray(subj)], X_num, X_denom)
    sim_data["estimate"] = p_to_bs(sim_data.prob, sim_data.beta, sim_data.N)
    sim_data["estimate"] = np.random.beta(sim_data.estimate*k, (1-sim_data.estimate)*k)


    
    return sim_data


def make_sim_data_bs_avg_d(n_participants, n_conditions, params, k = 20.):
    ## simulate average trial-level responses with a fixed amount of noise (untested 8/9/22, 1:24 PM)
    ## simulates bayesian sampler predictions (or simple conditional PTN judgments)
    
    all_thetas = make_thetas(n_participants)

    trial = list()
    cond = list()

    subj  = list()

    for p in range(0, n_participants):
            for c in range(0, n_conditions):
                for t in trial_types:
                    trial.append(t)
                    cond.append(c)

                    subj.append(p)
                    
    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])

    sim_data = pd.DataFrame(
            data = {
                "ID": subj,
                "querytype": trial, 
                "condition": cond
            }) >> s.mutate(
                conjdisj_trial = _.querytype.apply(is_conjdisj),
                d_base = _.ID.apply(lambda x: params["d_base"][x]),
                d_delta = _.ID.apply(lambda x: params["d_delta"][x]),
                theta = _.ID.apply(lambda x: all_thetas[x]),
                k = _.ID.apply(lambda x: params["k"][x]),
                d = _.d_base + _.d_delta * _.conjdisj_trial,
            )

    sim_data["prob"] = calc_prob(np.asarray(all_thetas)[np.asarray(subj)], X_num, X_denom)
    sim_data["estimate"] = p_to_ptn(sim_data.prob, sim_data.d)
    sim_data["estimate"] = np.random.beta(sim_data.estimate*k, (1-sim_data.estimate)*k)


    
    return sim_data


# def calc_ptnprob(trial, theta, d):
    
#     if is_cond(trial):
#         X_num = num_vecs[trial]
#         X_denom = denom_vecs[trial]
#         X_A = pA_vecs[trial]
        
#         p_numerator = np.sum(theta*X_num, axis=-1) #  = P(A & B)
#         pA = np.sum(theta*X_A, axis=-1)
#         num_or_denom = np.logical_or(X_A, X_denom).astype("float32")
#         p_num_or_denom = np.sum(theta * num_or_denom , axis=-1) # = P(A or B)


#         p_denom = np.sum(theta*X_denom, axis=-1)
#         numerator = (p_numerator)*(1-2*d)**2 + d*(1-2*d)*(pA + p_denom) + d**2
#         denom = ((1 - 2*d)*p_denom  + d)
#         p_ptn = numerator/denom

#         return p_ptn

#     else:
#         X_num = num_vecs[trial]
#         X_denom = denom_vecs[trial]

#         numerator = np.sum(theta*X_num, axis=-1)
#         denom = np.sum(theta*X_denom, axis=-1)
#         pi = np.divide(numerator, denom)
#         p_ptn = (1 - 2*d)*pi  + d

#         return p_ptn

    
def make_sim_data_ptn_avg(n_participants, n_conditions, params, simple_prob=False):

    all_thetas = make_thetas(n_participants)

    trial = list()
    cond = list()
    subj  = list()

    for p in range(0, n_participants):
            for c in range(0, n_conditions):
                for t in trial_types:
                    trial.append(t)
                    cond.append(c)
                    subj.append(p)
                    
    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    X_A = jnp.stack([pA_vecs[i] for i in trial])

    d = pd.DataFrame(
            data = {
                "ID": subj,
                "querytype": trial, 
                "condition": cond
            }) >> s.mutate(
                conjdisj_trial = _.querytype.apply(is_conjdisj),
                conditional_trial = _.querytype.apply(is_cond),
                d_base = _.ID.apply(lambda x: params["d_base"][x]),
                d_delta = _.ID.apply(lambda x: params["d_delta"][x]),
                theta = _.ID.apply(lambda x: all_thetas[x]),
                k = _.ID.apply(lambda x: params["k"][x])
            ) >> s.mutate(
                d = _.d_base + _.d_delta * _.conjdisj_trial
            )

    theta_vec = np.asarray(all_thetas)[np.asarray(subj)]

    if simple_prob:
        d["prob"] = calc_prob(theta_vec, X_num, X_denom)
    else:
        d["prob"] = prob_judge_PTN(np.asarray(d.querytype), theta_vec, X_num, X_denom, X_A, np.asarray(d.d))

    d["estimate"] = np.random.beta(np.asarray(d.prob)*np.asarray(d.k), (1-np.asarray(d.prob))*np.asarray(d.k))
    
    return d


def make_sim_data_ptn_trials(n_participants, n_blocks, n_conditions, params, simple_prob=False):
    all_thetas = make_thetas(n_participants)

    trial = list()
    cond = list()
    block = list()
    subj  = list()

    for p in range(0, n_participants):
        for b in range(0, n_blocks):
            for c in range(0, n_conditions):
                for t in trial_types:
                    trial.append(t)
                    cond.append(c)
                    block.append(b)
                    subj.append(p)
                    
    # setup "design matrix" (of sorts)
    X_num, X_denom = jnp.stack([num_vecs[i] for i in trial]), jnp.stack([denom_vecs[i] for i in trial])
    X_A = jnp.stack([pA_vecs[i] for i in trial])

    d = pd.DataFrame(
            data = {
                "ID": subj,
                "querytype": trial, 
                "block": block,
                "condition": cond
            }) >> s.mutate(
                conjdisj_trial = _.querytype.apply(is_conjdisj),
                d_base = _.ID.apply(lambda x: params["d_base"][x]),
                d_delta = _.ID.apply(lambda x: params["d_delta"][x]),
                theta = _.ID.apply(lambda x: all_thetas[x]),
                k = _.ID.apply(lambda x: params["k"][x])
            ) >> s.mutate(
                d = _.d_base + _.d_delta * _.conjdisj_trial
            )

    theta_vec = np.asarray(all_thetas)[np.asarray(subj)]

    if simple_prob:
        d["prob"] = calc_prob(theta_vec, X_num, X_denom)
    else:
        d["prob"] = prob_judge_PTN(np.asarray(d.querytype), theta_vec, X_num, X_denom, X_A, np.asarray(d.d))

    d["estimate"] = sim_ptn_samp_trial(np.asarray(d.prob), np.asarray(d.d),  np.asarray(d.k)) # 8/15/22, 6:33 PM choices to be made about N for PTN sim
    
    return d



def plot_param_recovery(posterior, params, param_name, var_name):
    x = (
    posterior >>
    s.mutate(d_base = _.d_base_pop + _.d_base_r * _.d_base_sd, 
             d_prime = _.d_base + np.exp(_.d_delta_pop + _.d_delta_r *  _.d_delta_sd)
#              d_prime = _.d_base + np.exp(_.d_delta_pop + _.d_delta_r *  _.d_delta_sd)
         ) 
    )

    x["param"] = x[param_name]
    x = (
        x >>
        s.group_by("ID") >>
        s.summarize(
        m_base = _.param.mean(),
        ll_base = _.param.quantile(.025),
        ul_base = _.param.quantile(.975),
        )
    )
    
    subj_params = (
        pd.DataFrame(params).reset_index() >> 
        s.rename(ID = _.index)
    )

    plt = (
        pd.merge(x, subj_params, on = "ID") >> 
        # s.mutate(ok = s.if_else(_.ll_base < _.d_base, s.if_else(_.ul_base > _.d_base, 1, 0), 0)) >>
        ggplot(aes(x=var_name, y = "m_base", ymin="ll_base", ymax="ul_base")) + 
        geom_pointrange() +
        geom_abline(intercept=0, slope=1, linetype="dashed") +
        # coord_flip() +
        theme(aspect_ratio=1)
    )

    return(plt)
