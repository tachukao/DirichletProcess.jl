"""Generate documents with LDA and HLDA
"""

module LDA
using Distributions
using StatsBase
using DirichletProcess

vocabulary = ["see", "spot", "run"]
n_terms = length(vocabulary)
mean_document_length = 5
term_dirichlet_parameter = 1
topic_dirichlet_parameter = 1

mutable struct Document
    topics::Array{Vector{Float64}}
    words::Array{String}
    len::Int
    Document() = new([], [], 0)
end

function add_word!(doc::Document, word, topic)
    push!(doc.topics, topic)
    push!(doc.words, word)
    doc.len += 1
    return nothing
end

function print_topic_counts(documents)
    for (i, doc) in enumerate(documents)
        println("Doc: $(i)")
        for (topic, count) in countmap(doc.topics)
            println("      count: $(count) | topic: $(round.(topic, digits=2))")
        end
    end
end

function finite(n_documents)
    n_topics = 2

    term_dirichlet_vector = ones(n_terms) * term_dirichlet_parameter
    term_distributions = [rand(Dirichlet(term_dirichlet_vector)) for _ in 1:n_topics]
    base_distribution = () -> rand(term_distributions)

    documents = [Document() for _ in 1:n_documents]

    for doc in documents
        topic_distribution = DP(base_distribution, topic_dirichlet_parameter)
        document_length = rand(Poisson(mean_document_length))
        for _ in 1:document_length
            p = dp_sample!(topic_distribution)
            word = vocabulary[rand(Categorical(p))]
            add_word!(doc, word, p)
        end
        println(doc.words)
    end

    print_topic_counts(documents)

    return nothing
end

function infinite(n_documents)
    base_dp_parameter = 10
    base_distribution = () -> rand(Dirichlet(term_dirichlet_parameter * ones(n_terms)))
    base_dp = DP(base_distribution, base_dp_parameter)

    documents = [Document() for _ in 1:n_documents]

    for doc in documents
        topic_distribution = DP(() -> dp_sample!(base_dp), topic_dirichlet_parameter)
        document_length = rand(Poisson(mean_document_length))
        for _ in 1:document_length
            p = dp_sample!(topic_distribution)
            word = vocabulary[rand(Categorical(p))]
            add_word!(doc, word, p)
        end
        println(doc.words)
    end

    print_topic_counts(documents)

    return nothing
end

end
