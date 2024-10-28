###
# From the results table, pick 2 agents and graph their trajectory in 3D state-space
# using the opinion, susceptibility, and gating states
# The 3D plot will show the trajectory of the agents in the state-space
# The trajectory will be plotted in 3D, with the x-axis representing the opinion state, the y-axis representing the susceptibility state, 
# and the z-axis representing the gating state
using WGLMakie
WGLMakie.activate!()

function plot_3dtrajectory(results, nb)
    # Pick nb agents
    agents = sample(1:length(unique(results.id)), nb, replace=false)
    println(agents)
    # Filter results for the selected agents
    results = filter(row -> row.id in agents, results)
    println(results[1:5,:])
    fig = Figure()
    Axis3(fig[1, 1], xlabel = "Opinion", ylabel = "Susceptibility", zlabel = "Gating")
    # Create color pattern (dictionary of agent id and their color from the color gradient)
    cmap = cgrad(:Spectral)
    endstep = maximum(results.time)
    enddata = filter(row -> row.time == endstep, results)
    enddata = sort(enddata, [:opinion_new])
    colororder = enddata.id # This is a list of agent id, in the order of their color gradient
    colorparts = Int(length(colororder)/nb)
    colororder = colororder[1:colorparts:end]
    colordict = Dict{Int, Int}() # agent id to color index
    for (i, id) in enumerate(colororder)
        colordict[id] = i
    end
    # Plot lines
    for grp in groupby(results, :id)
        # Pick color gradient based in dictionary
        agentcolor = cmap[colordict[grp.id[1]]/nb]
        lines!(grp.opinion_new, grp.scp_state, grp.gating_state, color = agentcolor, linewidth = 2)
    end
    scatter!(0,0,0, color = :red2, markersize = 20)
    return fig
end
# cmap = cgrad(:lightrainbow), color = cmap[grp.id[1]/100]
# agentcolor = cmap[grp.opinion_new[end] + 0.5]


