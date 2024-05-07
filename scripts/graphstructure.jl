using Graphs
using Random
using GraphMakie
using NetworkLayout



budparam = 3
seed = 33
g1 = erdos_renyi(numagents, numagents*budparam, seed = seed)
g2 = barabasi_albert(numagents, budparam, seed = seed)
g3 = watts_strogatz(numagents, budparam*2, 0.1, seed = seed)

function mygraphplot(g)
    f, ax, p = graphplot(g; #layout=Spring(dim=2, seed=5),
                        node_marker=Circle,
                        #nlabels=repr.(vertices(g)),
                        #nlabels_color=:black, nlabels_fontsize=11,
                        node_size=[sqrt(degree(g, i))*3 for i in vertices(g)], 
                        node_color=[:red4 for i in vertices(g)], 
                        edge_color=:grey32, line_width=0.5)
    hidedecorations!(ax); hidespines!(ax)
    ax.aspect = DataAspect()
    f
end

f1 = mygraphplot(g1)
f2 = mygraphplot(g2)
f3 = mygraphplot(g3)

save("plots/g_erdosrenyi_n$(numagents)_p$(budparam)_s$(seed).png", f1)
save("plots/g_barabasialbert_n$(numagents)_m$(budparam)_s$(seed).png", f2)
save("plots/g_wattsstrogatz_n$(numagents)_k$(budparam)_p01_s$(seed).png", f3)


# Plot histogram of degree distribution
using CairoMakie
fig1 = Figure(size = (300,300))
ax = Axis(fig1[1, 1], xlabel = "Number of agents", ylabel = "Degree distribution", yticks = minimum(degree(g1)):maximum(degree(g1)))
hist!(ax, degree(g1), bins=minimum(degree(g1))-0.5:1:maximum(degree(g1))+0.5, 
      #bar_labels = :values, label_formatter = x -> Int(round(x)), label_size = 15,
      direction=:x, offset = 0.0,
      strokewidth = 1, strokecolor = (:black, 0.5), color=:values)
hidespines!(ax)
fig1

fig2 = Figure(size = (300,300))
ax = Axis(fig2[1, 1], xlabel = "Number of agents", ylabel = "Degree distribution", yticks = minimum(degree(g2)):4:maximum(degree(g2)))
hist!(ax, degree(g2), bins=minimum(degree(g2))-0.5:4:maximum(degree(g2))+0.5, 
      #bar_labels = :values, label_formatter = x -> Int(round(x)), label_size = 15,
      direction=:x, offset = 0.0,
      strokewidth = 1, strokecolor = (:black, 0.5), color=:values)
hidespines!(ax)
fig2

fig3 = Figure(size = (300,300))
ax = Axis(fig3[1, 1], xlabel = "Number of agents", ylabel = "Degree distribution", yticks = minimum(degree(g3)):maximum(degree(g3)))
hist!(ax, degree(g3), bins=minimum(degree(g3))-0.5:1:maximum(degree(g3))+0.5, 
      #bar_labels = :values, label_formatter = x -> Int(round(x)), label_size = 15,
      direction=:x, offset = 0.0,
      strokewidth = 1, strokecolor = (:black, 0.5), color=:values)
hidespines!(ax)
fig3

save("plots/g_erdosrenyi_hist_degree_n$(numagents)_p$(budparam)_s$(seed).png", fig1)
save("plots/g_barabasialbert_hist_degree_n$(numagents)_m$(budparam)_s$(seed).png", fig2)
save("plots/g_wattsstrogatz_hist_degree_n$(numagents)_k$(budparam)_p01_s$(seed).png", fig3)






# Print some basic statistics about the resulting social graph buddies
println("Social graph initialized with $(jz_network[1]) network.")
println("Number of edges: $(ne(buddies))")
avgdeg = mean(Graphs.degree(buddies))
println("Average degree: $(avgdeg)")
println("Maximum degree: $(maximum(Graphs.degree(buddies)))")
println("Edge density: $(Graphs.density(buddies))")
println("Assortativity coef: $(assortativity(buddies))")
println("Global clustering coef: $(global_clustering_coefficient(buddies))")
println("Rich-club coef, k=mean: $(rich_club(buddies, Int(round(avgdeg))))")
println("Vertices with self-loops? $(has_self_loops(buddies))")
println("Graph is connected? $(is_connected(buddies))")