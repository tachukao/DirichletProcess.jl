module MixtureGibbs
using Distributions
using Plots
using StatsBase

function generate(n_samples)
    mixing = [0.2, 0.5, 0.3]
    model = MixtureModel(map(θ -> Normal(θ, 1.0), [-4.5, 0.0, 3.5]), mixing)
    return rand(model, n_samples)
end

mutable struct SS
    μ::Float64
    count::Int
end

struct HyperParameters
    α::Vector{Float64}
    λ1::Float64
    λ2::Float64
end

function sufficient_statistics(data, assignment)
    @assert length(data) == length(assignment)
    cluster_ids = unique(assignment)
    results = Dict{Int,SS}()
    for cluster_id in cluster_ids
        xs = data[assignment .== cluster_id]
        μ = mean(xs)
        count = length(xs)
        results[cluster_id] = SS(μ, count)
    end
    return results
end

function update_suffstats!(state)
    state.suffstats = sufficient_statistics(state.data, state.assignment)
    return nothing
end

mutable struct GibbState
    data::Vector{Float64}
    assignment::Vector{Int}
    cluster_ids::Vector{Int}
    π::Vector{Float64}
    θ::Vector{Float64}
    v::Float64
    hp::HyperParameters
    suffstats::Dict{Int,SS}

    function GibbState(data, n_clusters)
        cluster_ids = Vector(1:n_clusters)
        assignment = [rand(cluster_ids) for _ in 1:length(data)]
        α = ones(n_clusters)
        π = α / sum(α)
        θ = randn(n_clusters)
        v = 0.5
        hp = HyperParameters(α, 0.0, 1.0)
        suffstats = sufficient_statistics(data, assignment)
        state = new(data, assignment, cluster_ids, π, θ, v, hp, suffstats)
        return state
    end
end

function log_assignment_score(data_id, cluster_id, state)
    """log p(z_i=k ,|, cdot)
    """
    x = state.data[data_id]
    θ = state.θ[cluster_id]
    var = state.v
    log_pi = log.(state.π[cluster_id])
    return log_pi + logpdf(Normal(θ, var), x)
end

function assigment_probs(data_id, state)
    """p(z_i=cid ,|, cdot) for cid in cluster_ids
    """
    scores = map(cid -> log_assignment_score(data_id, cid, state), state.cluster_ids)
    scores = exp.(scores)
    return scores / sum(scores)
end

function sample_assignment(data_id, state)
    """Sample cluster assignment for data_id given current state

    cf Step 1 of Algorithm 2.1 in Sudderth 2006
    """
    p = assigment_probs(data_id, state)
    return state.cluster_ids[rand(Categorical(p))]
end

function update_assignment!(state)
    """Update cluster assignment for each data point given current state

    cf Step 1 of Algorithm 2.1 in Sudderth 2006
    """
    for (data_id, x) in enumerate(state.data)
        state.assignment[data_id] = sample_assignment(data_id, state)
        update_suffstats!(state)
    end
end

function sample_mixture_weights(state::GibbState)
    """Sample new mixture weights from current state according to
    a Dirichlet distribution

    cf Step 2 of Algorithm 2.1 in Sudderth 2006
    """
    ss = state.suffstats
    n_clusters = length(state.cluster_ids)
    alpha = [ss[cid].count + state.hp.α[cid] / n_clusters for cid in state.cluster_ids]
    return rand(Dirichlet(alpha))
end

function update_mixture_weights!(state::GibbState)
    """Update state with new mixture weights from current state
    sampled according to a Dirichlet distribution

    cf Step 2 of Algorithm 2.1 in Sudderth 2006
    """
    state.π = sample_mixture_weights(state)
    return nothing
end

function sample_cluster_mean(cluster_id, state)
    cluster_var = state.v
    hp_mean = state.hp.λ1
    hp_var = state.hp.λ2
    ss = state.suffstats[cluster_id]

    numerator = hp_mean / hp_var + ss.μ * ss.count / cluster_var
    denominator = (1.0 / hp_var + ss.count / cluster_var)
    posterior_mu = numerator / denominator
    posterior_var = 1.0 / denominator

    return randn() * sqrt(posterior_var) + posterior_mu
end

function update_cluster_means!(state::GibbState)
    state.θ = [sample_cluster_mean(cid, state) for cid in state.cluster_ids]
    return nothing
end

function gibb_step!(state::GibbState)
    update_assignment!(state)
    update_mixture_weights!(state)
    return update_cluster_means!(state)
end

function plot_cluster!(state::GibbState; opts...)
    data = state.data
    assignment = state.assignment
    histogram!(data; group=assignment, alpha=0.5, opts...)
    ylims!(0, 150)
    xlims!(-8, 8)
    return nothing
end

function run()
    n_clusters = 3
    data = generate(1000)
    state = GibbState(data, n_clusters)

    n_steps = 5
    fig = plot(; legend=false, layout=(n_steps, 1), size=(400, 600))
    for i in 1:n_steps
        plot_cluster!(state; subplot=i)
        annotate!(-5, 120, "step = $((i-1))"; subplot=i)
        gibb_step!(state)
    end
    display(fig)

    return nothing
end

function animation()
    n_clusters = 3
    data = generate(1000)
    state = GibbState(data, n_clusters)

    anim = @animate for i in 1:50
        plot(; legend=false)
        plot_cluster!(state)
        annotate!(-5, 120, "step = $((i-1))")
        gibb_step!(state)
    end

    return gif(anim, "assets/mixture_gibbs.gif"; fps=1)
end

end
