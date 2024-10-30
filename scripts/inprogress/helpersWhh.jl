using Random
using LinearAlgebra
using Distributions
using Graphs
using SimpleWeightedGraphs
using GraphMakie
using NetworkLayout
using CairoMakie
CairoMakie.activate!()

numagents = 100
seed = 332

function init_socgraph(type::String, numagents::Int, param::Int = 3, seed::Int = Int(round(time())))
    if type == "Erdos-Renyi"
        socgraph = erdos_renyi(numagents, numagents*param, seed = seed)
    elseif type == "Barabasi-Albert"
        socgraph = barabasi_albert(numagents, param, seed = seed)
    elseif type == "Watts-Strogatz"
        socgraph = watts_strogatz(numagents, param*2, 0.1, seed = seed)
    else
        println("Unknown graph type. Exiting.")
        return
    end
    println("Social graph initialized with $(type) network.")
    return socgraph
end

function adj_to_graph(A::AbstractMatrix; threshold::Float64=1e-3)
    if threshold > 0
        A = A .* (A .> threshold)
    end
    g = issymmetric(A) ? SimpleWeightedGraph(A) : SimpleWeightedDiGraph(A)
    return g
end

graph_to_adj(g; alpha::Float64=1.0) = adjacency_matrix(g) + alpha*I(ne(g))

function print_socgraph_descr(g)
    println("Number of edges: $(ne(g))")
    avgdeg = mean(Graphs.degree(g))
    println("Average degree: $(avgdeg)")
    println("Maximum degree: $(maximum(Graphs.degree(g)))")
    println("Edge density: $(Graphs.density(g))")
    println("Assortativity coef: $(assortativity(g))")
    println("Global clustering coef: $(global_clustering_coefficient(g))")
    println("Rich-club coef, k=mean: $(rich_club(g, Int(round(avgdeg))))")
    println("Vertices with self-loops? $(has_self_loops(g))")
    println("Graph is connected? $(is_connected(g))")
end

function plot_socgraph(g)
    fig, ax, p = graphplot(
        g; #layout=Spring(dim=2, seed=5),
        node_marker=Circle,
        #nlabels=repr.(vertices(g)),
        #nlabels_color=:black, nlabels_fontsize=11,
        node_size=[sqrt(degree(g, i))*3 for i in vertices(g)], 
        node_color=[:red4 for i in vertices(g)], 
        edge_color=:grey32, line_width=0.5)
    hidedecorations!(ax); hidespines!(ax)
    ax.aspect = DataAspect()
    return fig
end

function plot_degree_distrib(g)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Number of agents", ylabel = "Degree distribution", #yticks = minimum(degree(g)):maximum(degree(g))
    )
    hist!(ax, degree(g), bins=minimum(degree(g))-0.5:1:maximum(degree(g))+0.5, 
          direction=:x, offset = 0.0,
          strokewidth = 1, strokecolor = (:black, 0.5), color=:values)
    hidespines!(ax)
    return fig
end
