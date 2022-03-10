module Run
using HierarchicalDirichletProcess
using Distributions
using Plots

function dp_samples(alpha, n)
    dp = DP(randn, alpha)
    samples = [dp_sample!(dp) for _ in 1:n]
    return samples
end

function hdp_samples(alpha1, alpha2, n)
    hdp = HDP(randn, alpha1, alpha2)
    samples = [hdp_sample!(hdp) for _ in 1:n]
    return samples
end

function dp()
    n = 10000
    fig = plot(; layout=(2, 2), legend=false)
    alphas = [1, 10, 100, 1000]
    for i in 1:4
        alpha = alphas[i]
        histogram!(dp_samples(alpha, n); subplot=i, normalize=true)
    end
    display(fig)

    return nothing
end

function hdp()
    n = 10000
    fig = plot(; layout=(2, 2), legend=false)
    # alphas = [1, 10, 100, 1000]
    alphas = [10, 10, 10, 10]
    for i in 1:4
        alpha = alphas[i]
        histogram!(hdp_samples(100, alpha, n); subplot=i, normalize=true)
    end
    display(fig)

    return nothing
end

end
