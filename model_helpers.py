import jax.numpy as jnp
import numpy as np


def is_conjdisj(trial):
    conjdisj = set(["AorB","notAorB","AornotB","notAornotB", "AandB", "notAandB", "AandnotB", "notAandnotB"])
    x = 1 if trial in conjdisj else 0
    return x

def load_data():
    import glob
    df = pd.concat([pd.read_csv(f) for f in glob.glob('osfstorage-archive/Experiment 2/*.csv')], ignore_index = True)

    fix_id_dict = dict()
    original_ids = list(np.unique(df.ID))

    for i in range(0, len(original_ids)):
        fix_id_dict[original_ids[i]] = i

    df = (df >> 
         filter_by(~X.querydetail.str.contains("warm|snowy")) >>
          mutate(estimate = X.estimate/100.) >>
          group_by(X.ID, X.querytype, X.querydetail) >>
          summarize(estimate = np.mean(X.estimate)) >>
          mutate(coldrainy = X.querydetail.str.contains("cold|rainy").astype("int")) >>
          mutate(estimate = X.estimate.replace({0:.01, 1:.99})) >>
          mutate(conjdisj_trial = X.querytype.apply(is_conjdisj)) >>
          mutate(ID = X.ID.apply(lambda x: fix_id_dict[x]))
         )
    
    return df

def make_model_data(data):
    df_test = data
    # df_test = df >> filter_by(X.conjdisj_trial==0)
    # df_test = df >> filter_by(X.coldrainy==1)

    trial_list = df_test.querytype
    X_num = jnp.stack([num_vecs[trial] for trial in trial_list])
    X_denom = jnp.stack([denom_vecs[trial] for trial in trial_list])

    X_data = {
        "trial": trial_list,
        "X_num": X_num,
        "X_denom": X_denom,
        "subj": jnp.array(list(df_test.ID)),
        "conjdisj_trial": jnp.array(list(df_test.conjdisj_trial)),
        "cond": jnp.array(list(df_test.coldrainy)).astype("int32")
    #     "block": jnp.array(list(df_test.block))
    }

    y_data = jnp.array(df_test.estimate.to_numpy())
    
    return X_data, y_data

trial_funcs = dict({
    "AandB": lambda theta: jnp.matmul(theta, jnp.array([1.,0.,0.,0.])) ,
    "AandnotB": lambda theta: jnp.matmul(theta, jnp.array([0.,1.,0.,0.])),
    "notAandB": lambda theta: jnp.matmul(theta, jnp.array([0.,0.,1.,0.])),
    "notAandnotB": lambda theta: jnp.matmul(theta, jnp.array([0.,0.,0.,1.])),
    "A":lambda theta: jnp.matmul(theta, jnp.array([1.,1.,0.,0.])),
    "B":lambda theta: jnp.matmul(theta, jnp.array([1.,0.,1.,0.])),
    "notA":lambda theta: jnp.matmul(theta, jnp.array([0.,0.,1.,1.])),
    "notB":lambda theta: jnp.matmul(theta, jnp.array([0.,1.,0.,1.])),
    "AorB":lambda theta: jnp.matmul(theta, jnp.array([1.,1.,1.,0.])),
    "AornotB":lambda theta: jnp.matmul(theta, jnp.array([1.,1.,0.,1.])),
    "notAorB":lambda theta: jnp.matmul(theta, jnp.array([0.,1.,1.,1.])),
    "notAornotB":lambda theta: jnp.matmul(theta, jnp.array([0.,1.,0.,1.])),
    
    "AgB": lambda theta: jnp.divide( jnp.matmul(theta, jnp.array([1.,0.,0.,0.])), jnp.matmul(theta, jnp.array([1.,0.,1.,0.])) ),
    "notAgB": lambda theta: jnp.divide( jnp.matmul(theta, jnp.array([0.,0.,1.,0.])), jnp.matmul(theta, jnp.array([1.,0.,1.,0.])) ),
    "AgnotB": lambda theta: jnp.divide( jnp.matmul(theta, jnp.array([0.,1.,0.,0.])), jnp.matmul(theta, jnp.array([0.,1.,0.,1.])) ),
    "notAgnotB": lambda theta: jnp.divide( jnp.matmul(theta, jnp.array([0.,0.,0.,1.])), jnp.matmul(theta, jnp.array([0.,1.,0.,1.])) ),
    "BgA": lambda theta: jnp.divide( jnp.matmul(theta, jnp.array([1.,0.,0.,0.])), jnp.matmul(theta, jnp.array([1.,1.,0.,0.])) ),
    "notBgA": lambda theta: jnp.divide( jnp.matmul(theta, jnp.array([0.,1.,0.,0.])), jnp.matmul(theta, jnp.array([1.,1.,0.,0.])) ),
    "BgnotA": lambda theta: jnp.divide( jnp.matmul(theta, jnp.array([0.,0.,1.,0.])), jnp.matmul(theta, jnp.array([0.,0.,1.,1.])) ),
    "notBgnotA": lambda theta: jnp.divide( jnp.matmul(theta, jnp.array([0.,0.,0.,1.])), jnp.matmul(theta, jnp.array([0.,0.,1.,1.])) )
})

def sim_sampling(p, beta, N, k):

    p_bs = p * N / (N + 2.*beta) + beta/(N + 2.*beta)
#     return np.random.normal(p_bs, k)
    return np.random.beta(p_bs*k, (1-p_bs)*k)

def calc_prob(trial, theta):
    ## compute implied subj. probability from latent theta and trial type
    ## this is a vectorized solution: https://bit.ly/2P6mMcD
    
    return trial_funcs[trial](theta)

# def sim_sampling(p, beta, N, k):

#     p_bs = p * N / (N + 2.*beta) + beta/(N + 2.*beta)
# #     return np.random.normal(p_bs, k)
#     return np.random.beta(p_bs*k, (1-p_bs)*k)


# def dm_probs(trial_data, theta, n_obs):

#     ## compute implied subj. probability from latent theta and trial type
#     ## this is a vectorized solution: https://bit.ly/2P6mMcD
#     p = jnp.ones(0)
#     for i in range(0, n_obs):
#         temp = trial_funcs[trial_data[i]](theta)
#         p = jnp.concatenate( (p, jnp.array([temp])), 0)

#     return p



def make_thetas(n):
    return [np.random.dirichlet(jnp.ones(4)) for _ in range(0,n)]

# def create_response_vec(trials, thetas, beta, N, k):
#     all_responses = jnp.ones(0)
#     for i in range(0, n_participants):
#         theta = thetas[i]

#         probs = dm_probs(trials, theta, len(trials))
#         responses = sim_sampling(probs, beta=beta, N=N, k=k)
#         all_responses = jnp.concatenate((all_responses, responses))
#     return all_responses


# def sim_sampling2(p, beta, N, sigma):
#     pi = np.random.beta(p*N, (1-p)*N)
#     p_bs = pi * N / (N + 2.*beta) + beta/(N + 2.*beta)
#     return np.random.normal(p_bs, sigma)


# def create_response_vec(trials, thetas, beta, N, k):
#     all_responses = jnp.ones(0)
#     for i in range(0, n_participants):
#         theta = thetas[i]

#         probs = dm_probs(trials, theta, len(trials))
#         responses = sim_sampling(probs, beta=beta, N=N, k =k)
#         all_responses = jnp.concatenate((all_responses, responses))
#     return all_responses


# def create_response_vec2(trials, thetas, beta, N, sigma):
#     all_responses = jnp.ones(0)
#     for i in range(0, n_participants):
#         theta = thetas[i]

#         probs = dm_probs(trials, theta, len(trials))
#         responses = sim_sampling2(probs, beta=beta, N=N, sigma=sigma)
#         all_responses = jnp.concatenate((all_responses, responses))
#     return all_responses


# def create_response_vec_complex(trials, thetas, beta, N_base, N_prime, k):
#     # for the "complex" version of the model
#     all_responses = jnp.ones(0)
    
#     N_delta = N_prime - N_base
#     Ns = {1:N_prime, 0:N_base}
    
#     for i in range(0, n_participants):
#         theta = thetas[i]

#         probs = dm_probs(trials, theta, len(trials))
#         conj_disj = [is_conjdisj(i) for i in trials]
#         N = jnp.array([Ns[i] for i in conj_disj])
#         responses = sim_sampling(probs, beta=beta, N=N, k=k)
#         all_responses = jnp.concatenate((all_responses, responses))
        
#     return all_responses


num_vecs = dict({
    "AandB": jnp.array([1.,0.,0.,0.]),
    "AandnotB": jnp.array([0.,1.,0.,0.]),
    "notAandB": jnp.array([0.,0.,1.,0.]),
    "notAandnotB": jnp.array([0.,0.,0.,1.]),
    "A":jnp.array([1.,1.,0.,0.]),
    "B":jnp.array([1.,0.,1.,0.]),
    "notA":jnp.array([0.,0.,1.,1.]),
    "notB":jnp.array([0.,1.,0.,1.]),
    "AorB":jnp.array([1.,1.,1.,0.]),
    "AornotB":jnp.array([1.,1.,0.,1.]),
    "notAorB":jnp.array([0.,1.,1.,1.]),
    "notAornotB":jnp.array([0.,1.,0.,1.]),
   
    "AgB": jnp.array([1.,0.,0.,0.]),
    "notAgB": jnp.array([0.,0.,1.,0.]),
    "AgnotB": jnp.array([0.,1.,0.,0.]),
    "notAgnotB": jnp.array([0.,0.,0.,1.]),
    "BgA": jnp.array([1.,0.,0.,0.]),
    "notBgA": jnp.array([0.,1.,0.,0.]),
    "BgnotA": jnp.array([0.,0.,1.,0.]),
    "notBgnotA": jnp.array([0.,0.,0.,1.])
})

denom_vecs = dict({
    "AandB":  jnp.array([1.,1.,1.,1.]),
    "AandnotB":  jnp.array([1.,1.,1.,1.]),
    "notAandB":  jnp.array([1.,1.,1.,1.]),
    "notAandnotB":  jnp.array([1.,1.,1.,1.]),
    "A": jnp.array([1.,1.,1.,1.]),
    "B": jnp.array([1.,1.,1.,1.]),
    "notA": jnp.array([1.,1.,1.,1.]),
    "notB": jnp.array([1.,1.,1.,1.]),
    "AorB": jnp.array([1.,1.,1.,1.]),
    "AornotB": jnp.array([1.,1.,1.,1.]),
    "notAorB": jnp.array([1.,1.,1.,1.]),
    "notAornotB": jnp.array([1.,1.,1.,1.]),
    
    "AgB": jnp.array([1.,0.,1.,0.]),
    "notAgB": jnp.array([1.,0.,1.,0.]),
    "AgnotB": jnp.array([0.,1.,0.,1.]),
    "notAgnotB": jnp.array([0.,1.,0.,1.]),
    "BgA": jnp.array([1.,1.,0.,0.]),
    "notBgA": jnp.array([1.,1.,0.,0.]),
    "BgnotA": jnp.array([0.,0.,1.,1.]),
    "notBgnotA": jnp.array([0.,0.,1.,1.])
})

def make_arviz_data(mcmc, model, data):
    posterior_samples = mcmc.get_samples()

    posterior_predictive = Predictive(model, posterior_samples)(
        random.PRNGKey(1), data
    )
    prior = Predictive(model, num_samples=500)(
        random.PRNGKey(2), data
    )

    return az.from_numpyro(
        mcmc,
        prior = prior,
        posterior_predictive = posterior_predictive
    )