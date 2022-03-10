module MixtureCollapsedGibbs
using Distributions
using Plots
using StatsBase

include("gibbs.jl")

abstract type ModelType end
abstract type Parametric <: ModelType end
abstract type NonParametric <: ModelType end

model_name(::Type{Parametric}) = "parametric"
model_name(::Type{NonParametric}) = "nonparametric"

function log_predictive_likelihood(::Type{Parametric}, data_id, cluster_id, state)
    """Predictive likelihood of the data at data_id is generated
    by cluster_id given the currenbt state.

    From Section 2.4 of
    http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    """
    ss = state.suffstats[cluster_id]
    hp_mean = state.hp.λ1
    hp_var = state.hp.λ2
    param_var = state.v
    x = state.data[data_id]
    return _helper_log_predictive_likelihood(ss, hp_mean, hp_var, param_var, x)
end

function _helper_log_predictive_likelihood(ss, hp_mean, hp_var, param_var, x)
    posterior_sigma2 = 1 / (ss.count * 1.0 / param_var + 1.0 / hp_var)
    predictive_mu =
        posterior_sigma2 * (hp_mean * 1.0 / hp_var + ss.count * ss.μ * 1.0 / param_var)
    predictive_sigma2 = param_var + posterior_sigma2
    predictive_sd = sqrt(predictive_sigma2)
    return logpdf(Normal(predictive_mu, predictive_sd), x)
end

function log_cluster_assign_score(::Type{Parametric}, cluster_id, state)
    """Log-likelihood that a new point generated will
    be assigned to cluster_id given the current state.
    """
    current_cluster_size = state.suffstats[cluster_id].count
    num_clusters = length(state.cluster_ids)
    α = state.hp.α
    return log(current_cluster_size + α * 1.0 / num_clusters)
end

function cluster_assignment_distribution(::Type{Parametric}, data_id, state)
    """Compute the marginal distribution of cluster assignment
    for each cluster.
    """
    labels = []
    scores = []
    for (cid, _) in state.suffstats
        score = log_predictive_likelihood(Parametric, data_id, cid, state)
        score += log_cluster_assign_score(Parametric, cid, state)
        score = exp(score)
        push!(labels, cid)
        push!(scores, score)
    end
    scores /= sum(scores)
    return labels, scores
end

function add_datapoint_to_suffstats(x, ss)
    """Add datapoint to sufficient stats for normal component
    """
    nμ = (ss.μ * (ss.count) + x) / (ss.count + 1)
    ncount = ss.count + 1
    return SS(nμ, ncount)
end

function remove_datapoint_from_suffstats(x, ss)
    """Remove datapoint from sufficient stats for normal component
    """
    nμ = (ss.μ * (ss.count) - x * 1.0) / (ss.count - 1)
    ncount = ss.count - 1
    return SS(nμ, ncount)
end

function gibb_step!(::Type{Parametric}, state::GibbsState)
    both = zip(state.data, state.assignment)
    for (data_id, (datapoint, cid)) in enumerate(both)
        state.suffstats[cid] = remove_datapoint_from_suffstats(
            datapoint, state.suffstats[cid]
        )
        labels, scores = cluster_assignment_distribution(Parametric, data_id, state)
        cid = labels[rand(Categorical(scores))]
        state.assignment[data_id] = cid
        state.suffstats[cid] = add_datapoint_to_suffstats(
            state.data[data_id], state.suffstats[cid]
        )
    end

    return nothing
end

function log_cluster_assign_score(::Type{NonParametric}, cluster_id, state)
    """Log-likelihood that a new point generated will
    be assigned to cluster_id given the current state.
    """
    if cluster_id == typemax(Int)
        return log(state.hp.α)
    else
        return log(state.suffstats[cluster_id].count)
    end
end

function log_predictive_likelihood(::Type{NonParametric}, data_id, cluster_id, state)
    """Predictive likelihood of the data at data_id is generated
    by cluster_id given the currenbt state.

    From Section 2.4 of
    http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    """
    if cluster_id == typemax(Int)
        ss = SS(0, 0)
    else
        ss = state.suffstats[cluster_id]
    end

    hp_mean = state.hp.λ1
    hp_var = state.hp.λ2
    param_var = state.v
    x = state.data[data_id]
    return _helper_log_predictive_likelihood(ss, hp_mean, hp_var, param_var, x)
end

function cluster_assignment_distribution(::Type{NonParametric}, data_id, state)
    """Compute the marginal distribution of cluster assignment
    for each cluster.
    """
    labels = []
    scores = []
    for cid in [keys(state.suffstats)..., typemax(Int)]
        score = log_predictive_likelihood(NonParametric, data_id, cid, state)
        score += log_cluster_assign_score(NonParametric, cid, state)
        score = exp(score)
        push!(labels, cid)
        push!(scores, score)
    end
    scores /= sum(scores)
    return labels, scores
end

function create_cluster!(state)
    cluster_id = maximum(keys(state.suffstats)) + 1
    state.suffstats[cluster_id] = SS(0, 0)
    push!(state.cluster_ids, cluster_id)
    return cluster_id
end

function destroy_cluster!(state, cluster_id)
    delete!(state.suffstats, cluster_id)
    filter!(x -> x != cluster_id, state.cluster_ids)
    return nothing
end

function prune_clusters!(state)
    for cid in state.cluster_ids
        if state.suffstats[cid].count == 0
            destroy_cluster!(state, cid)
        end
    end
end

function sample_assignment(data_id, state)
    """Sample new assignment from marginal distribution.
    If cluster is "`new`", create a new cluster.
    """
    labels, scores = cluster_assignment_distribution(NonParametric, data_id, state)
    cid = labels[rand(Categorical(scores))]
    if cid == typemax(Int)
        return create_cluster!(state)
    else
        return cid
    end
end

function gibb_step!(::Type{NonParametric}, state)
    """Collapsed Gibbs sampler for Dirichlet Process Mixture Model
    """
    both = zip(state.data, state.assignment)
    for (data_id, (datapoint, cid)) in enumerate(both)
        state.suffstats[cid] = remove_datapoint_from_suffstats(
            datapoint, state.suffstats[cid]
        )
        prune_clusters!(state)
        cid = sample_assignment(data_id, state)
        state.assignment[data_id] = cid
        state.suffstats[cid] = add_datapoint_to_suffstats(
            state.data[data_id], state.suffstats[cid]
        )
    end
end

function plot_cluster!(state::GibbsState; opts...)
    data = state.data
    assignment = state.assignment
    histogram!(data; group=assignment, alpha=0.5, opts...)
    ylims!(0, 150)
    xlims!(-8, 8)
    return nothing
end

function run(T, n_clusters)
    Random.seed!(0)
    data = generate(1000)
    state = GibbsState(data, n_clusters; α=0.01)
    n_steps = 5
    fig = plot(; legend=false, layout=(n_steps, 1), size=(400, 600))
    for i in 1:n_steps
        plot_cluster!(state; subplot=i)
        annotate!(-5, 120, "step = $((i-1))"; subplot=i)
        gibb_step!(T, state)
    end
    display(fig)

    return nothing
end

function animation(T, n_clusters)
    Random.seed!(0)
    data = generate(1000)
    state = GibbsState(data, n_clusters; α=0.1)

    anim = @animate for i in 1:50
        plot(; legend=false)
        plot_cluster!(state)
        annotate!(-5, 120, "step = $((i-1))")
        gibb_step!(T, state)
    end

    return gif(anim, "assets/$(model_name(T))_mixture_collapsed_gibbs.gif"; fps=1)
end

end
