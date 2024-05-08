# This script contains a function that plots the social graph at the end of the simulation, 
# with node colors corresponding to opinion states of agents and node sizes corresponding to their degree.

using GraphMakie
using NetworkLayout

function viz_socgraph(g, res, dim=2)
    # Create color pattern (dictionary of agent id and their color from the color gradient)
    cmap = cgrad(:Spectral)
    endstep = maximum(res.time)
    enddata = filter(row -> row.time == endstep, res)
    enddata = sort(enddata, [:opinion_new])
    nodeorder = enddata.id # This is a list of node id, in the order of their color gradient
    nodespercolor = nv(g)/length(cmap)
    groupsize = ceil(Int, nodespercolor)
    # Create vector of colors of all nodes
    nodecolors = []
    for i in 1:length(nodeorder)
        push!(nodecolors, cmap[ceil(Int, i/groupsize)])
    end
    nodecolors
    # Add nodecolors as a column to enddata
    enddata[!,:nodecolor] = nodecolors
    # Sort again by id
    sortedenddata = sort(enddata, [:id])

    # Plot the social graph, with node colors corresponding to opinion states
    f, ax, p = graphplot(g; layout=Spring(dim=dim, C=5.0),
                        node_marker=Circle, 
                        #nlabels=repr.(vertices(g)),
                        #nlabels_color=:black, nlabels_fontsize=11,
                        node_size=[sqrt(degree(g, i))*3 for i in vertices(g)], 
                        node_color=[sortedenddata.nodecolor[i] for i in vertices(g)], 
                        edge_color=:grey32, line_width=0.3)
    if dim == 2
        hidedecorations!(ax); hidespines!(ax)
        ax.aspect = DataAspect()
    end
    return f, enddata, sortedenddata
end