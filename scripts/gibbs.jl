function generate(n_samples)
    mixing = [0.2, 0.5, 0.3]
    model = MixtureModel(map(θ -> Normal(θ, 1.0), [-4.5, 0.0, 4.]), mixing)
    return rand(model, n_samples)
end

mutable struct SS
    μ::Float64
    count::Int
end

struct HyperParameters
    α::Float64
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

mutable struct GibbsState
    data::Vector{Float64}
    assignment::Vector{Int}
    cluster_ids::Vector{Int}
    π::Vector{Float64}
    θ::Vector{Float64}
    v::Float64
    hp::HyperParameters
    suffstats::Dict{Int,SS}

    function GibbsState(data, n_clusters; α=1)
        cluster_ids = Vector(1:n_clusters)
        assignment = [rand(cluster_ids) for _ in 1:length(data)]
        π = ones(n_clusters) / n_clusters
        θ = randn(n_clusters)
        v = 1.
        hp = HyperParameters(α, 0.0, 1.0)
        suffstats = sufficient_statistics(data, assignment)
        state = new(data, assignment, cluster_ids, π, θ, v, hp, suffstats)
        return state
    end
end
