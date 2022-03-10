module DirichletProcess

using Distributions

export DP, dp_sample!
export HDP, hdp_sample!

# Write your package code here.
mutable struct DP
    base_measure::Function
    alpha::Float64
    cache
    weights
    total_stick_used::Float64
    DP(base_measure, alpha) = new(base_measure, alpha, [], [], 0.0)
end

mutable struct HDP
    base_measure::Function
    alpha1::Float64
    alpha2::Float64
    dp1::DP
    dp2::DP
    function HDP(base_measure, alpha1, alpha2)
        dp1 = DP(base_measure, alpha1)
        dp2 = DP(() -> dp_sample!(dp1), alpha2)
        return new(base_measure, alpha1, alpha2, dp1, dp2)
    end
end

function dp_sample!(dp::DP)
    remaining = 1.0 - dp.total_stick_used
    i = rand(Categorical([dp.weights..., remaining]))
    if i <= length(dp.weights)
        return dp.cache[i]
    else
        stick_piece = rand(Beta(1, dp.alpha)) * remaining
        dp.total_stick_used += stick_piece
        push!(dp.weights, stick_piece)
        new_value = dp.base_measure()
        push!(dp.cache, new_value)
        return new_value
    end
end

function hdp_sample!(hdp::HDP)
    return dp_sample!(hdp.dp2)
end

end
