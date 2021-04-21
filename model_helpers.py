import jax.numpy as jnp
import numpy as np


### ------ Data processing

def load_raw_data(x=1):
    import glob
    df = pd.concat([pd.read_csv(f) for f in glob.glob('osfstorage-archive/Experiment {}/*.csv'.format(x))], ignore_index = True)

    original_ids = list(np.unique(df.ID))
    fix_id_dict = {original_ids[i]:i for i in range(0, len(original_ids))}

#     df = (df >> 
#           mutate(estimate = X.estimate/100.) >>
#           mutate(ID = X.ID.apply(lambda x: fix_id_dict[x]))
#          )
    df = (df.assign(estimate = df.estimate/100., ID=df.ID.apply(lambda x: fix_id_dict[x]))
         )
    
    return df


def load_data_exp1():
    df = load_raw_data(1)
    df["condition"] = np.select([df.querydetail.str.contains("icy|frosty")],[0], default=0)

    return df


def load_data_exp2():
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

    return df

def load_query_avg_data_exp2():

    df = (load_data_exp2()
          .groupby(["ID","condition","querytype","querydetail"], as_index=False)
          .agg({"estimate":"mean"})
          .assign(estimate = lambda x: x.estimate.replace({0:.01, 1:.99}))
         )
    
    return df

def load_query_avg_data_exp1():

    df = (load_data_exp1()
          .groupby(["ID", "condition", "querytype", "querydetail"], as_index=False)
          .agg({"estimate":"mean"})
          .assign(estimate = lambda x: x.estimate.replace({0:.01, 1:.99}))
         )
    
    return df


def load_query_avg_data_all():
    # this won't work well for model fitting b/c different numbers of conditions
    # will break my manual indexing
    df1 = load_query_avg_data_exp1().assign(exp=1)
    df2 = load_query_avg_data_exp2().assign(exp=2)
    
    df = pd.concat([df1, df2.assign(ID=df2["ID"]+5000)], ignore_index=True) # bit of a kludge
    
    original_ids = list(np.unique(df.ID))
    fix_id_dict = {original_ids[i]:i for i in range(0, len(original_ids))}
    
    df = df.assign(ID = df.ID.apply(lambda x: fix_id_dict[x]))
    
    return df
    

def make_model_data(data):

    X_data = {
        "trial": data.querytype,
        "subj": jnp.array(list(data.ID)),
        "cond": jnp.array(list(data.condition), dtype="int32"),
    }

    y_data = jnp.array(data.estimate.to_numpy())
    
    return X_data, y_data


### -------- Simulation functions

def sim_sampling(p, beta, N, k):

    p_bs = p * N / (N + 2.*beta) + beta/(N + 2.*beta)
#     return np.random.normal(p_bs, k)
    return np.random.beta(p_bs*k, (1-p_bs)*k)

def calc_prob(trial, theta):
    ## compute implied subj. probability from latent theta and trial type
    ## this is a vectorized solution: https://bit.ly/2P6mMcD
    
    return trial_funcs[trial](theta)

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



def sim_bayesian_sampler(trial_types, n_participants, n_blocks, params):

    n_trial_types = len(trial_types)
    trials = trial_types*n_blocks
    conds = np.tile(np.array([0,1]).repeat(len(trial_types)), n_participants)
    blocks = list(np.repeat(np.array(list(range(0, n_blocks))), n_trial_types))

    all_participants = list(np.repeat(np.array(list(range(0, n_participants))), n_trial_types*n_blocks))
    all_thetas = make_thetas(n_participants)
    all_trials = trials*n_participants
    all_blocks = blocks*n_participants

    sim_data = pd.DataFrame(
        data = {
            "ID": all_participants,
            "querytype": all_trials, 
            "block": all_blocks,
            "coldrainy": conds
        }) >> mutate(
            conjdisj_trial = X.querytype.apply(is_conjdisj),
            N_base = X.ID.apply(lambda x: params["N_base"][x]),
            N_delta = X.ID.apply(lambda x: params["N_delta"][x]),
            beta = X.ID.apply(lambda x: params["beta"][x]),
        theta = X.ID.apply(lambda x: all_thetas[x])
        ) >> mutate(N = X.N_base + X.N_delta * abs(1-X.conjdisj_trial))

    sim_data["prob"] = sim_data.apply(lambda x: calc_prob(x.querytype, x.theta), axis=1)
    sim_data["estimate"] = sim_data.apply(lambda x: sim_sampling(x.prob, x.beta, x.N, params["k"]), axis=1)
    
    return sim_data


def make_thetas(n):
    return [np.random.dirichlet(jnp.ones(4)) for _ in range(0,n)]


#### ---- model construction / design matrix construction

def sigmoid(x):  
    return jnp.exp(-jnp.logaddexp(0, -x)) # numerically stable


def is_cond(trial):
    return float(bool(re.search("g",trial)))


def is_conjdisj(trial):
    conjdisj = set(["AorB","notAorB","AornotB","notAornotB", "AandB", "notAandB", "AandnotB", "notAandnotB"])
    x = 1 if trial in conjdisj else 0
    return x


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


pA_vecs = dict({
    "AandB": jnp.ones(4), # unused
    "AandnotB": jnp.ones(4),
    "notAandB": jnp.ones(4),
    "notAandnotB": jnp.ones(4),
    "A":jnp.ones(4),
    "B":jnp.ones(4),
    "notA":jnp.ones(4),
    "notB":jnp.ones(4),
    "AorB":jnp.ones(4),
    "AornotB":jnp.ones(4),
    "notAorB":jnp.ones(4),
    "notAornotB":jnp.ones(4),
   
    "AgB": jnp.array([1.,1.,0.,0.]),
    "notAgB": jnp.array([0.,0.,1.,1.]),
    "AgnotB": jnp.array([1.,1.,0.,0.]),
    "notAgnotB": jnp.array([0.,0.,1.,1.]),
    "BgA": jnp.array([1.,0.,1.,0.]),
    "notBgA": jnp.array([0.,1.,0.,1.]),
    "BgnotA": jnp.array([1.,0.,1.,0.]),
    "notBgnotA": jnp.array([0.,1.,0.,1.])
})

### ------ Arviz

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


def count_divergences(model_data):
    return np.sum(model_data.sample_stats.diverging.values)


### ---- plotting


def plot_model_preds(orig_data, model_data):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].set_xlim(0,1)
    axes[1].set_xlim(0,1)
    axes[0].set_ylim(0,1)
    axes[1].set_ylim(0,1)
    axes[0].set_aspect(1)
    axes[1].set_aspect(1)
#     fig.suptitle('Model')

    d = orig_data
    d["preds"] = model_data.posterior_predictive.mean(dim=['chain', 'draw']).yhat
    print("Participant mean response corr = ", np.round(np.corrcoef(d.estimate, d.preds)[0,1],3))
    sns.scatterplot(ax = axes[0], x = d.preds, y = d.estimate)

    trial_df = (d
                .groupby(["querytype", "querydetail"])
                .agg({"preds":"mean", "estimate":"mean"})
               )
    print("Query-level corr", np.round(np.corrcoef(trial_df.estimate, trial_df.preds)[0,1],3))
    sns.scatterplot(ax = axes[1], x = trial_df.preds, y = trial_df.estimate)
    az.plot_ppc(ax=axes[2], data=model_data, data_pairs={"yhat":"yhat"},num_pp_samples=500, legend=False)
    axes[0].set_title("Avg. participant-level responses")    
    axes[1].set_title("Avg. responses for each query")
    axes[2].set_title("posterior predictive")