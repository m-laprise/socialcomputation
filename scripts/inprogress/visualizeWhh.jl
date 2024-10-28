using Graphs
using Random
using GraphMakie
using NetworkLayout
using LinearAlgebra
using Distributions

using CairoMakie
CairoMakie.activate!()

numagents = 100
seed = 332

A = adjacency_matrix(socgraph) + 1I(numagents)

function initialize_socgraph(type::String, numagents::Int, param::Int = 3, seed::Int = Int(round(time())))
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
        g; layout=Spring(dim=2, seed=5),
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
    fig = Figure(size = (300,300))
    ax = Axis(fig[1, 1], xlabel = "Number of agents", ylabel = "Degree distribution", yticks = minimum(degree(g)):maximum(degree(g)))
    hist!(ax, degree(g), bins=minimum(degree(g))-0.5:1:maximum(degree(g))+0.5, 
          #bar_labels = :values, label_formatter = x -> Int(round(x)), label_size = 15,
          direction=:x, offset = 0.0,
          strokewidth = 1, strokecolor = (:black, 0.5), color=:values)
    hidespines!(ax)
    return fig
end

f1 = mygraphplot(socgraph)
f2 = plot_degree_distrib(socgraph)

