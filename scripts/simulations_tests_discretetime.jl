using DrWatson
using JLD2

include("BFL_discretetime_mod.jl")
include("noise_and_input_patterns.jl")

#@quickactivate "socialcomputation"

# Create function that passes any kwargs to model_run, run simulations and generate figure
function runsim_oneparam(numagents, maxsteps, inputmatrix, noisematrix, gating=false, seed=25; kwargs...)
    if gating
        varstocollect = [:opinion_new, :scp_state, :gating_state]
    else
        varstocollect = [:opinion_new, :scp_state]
    end
    data, graph = model_run([:opinion_new, :scp_state, :gating_state]; 
                    numagents = numagents, 
                    maxsteps = maxsteps, 
                    inputmatrix = inputmatrix, 
                    noisematrix = noisematrix,
                    gating = gating, 
                    seed = seed, 
                    kwargs...)
    results = DataFrame(data)
    return results, graph
end

function runsim_oneparam_g(numagents, maxsteps, inputmatrix, noisematrix, seed=25; kwargs...)
    nog_data, graph = model_run([:opinion_new, :scp_state]; 
                    numagents = numagents, 
                    maxsteps = maxsteps, 
                    inputmatrix = inputmatrix, 
                    noisematrix = noisematrix,
                    gating = false, 
                    seed = seed, 
                    kwargs...)
    nog_results = DataFrame(nog_data)
    wg_data, graph = model_run([:opinion_new, :scp_state, :gating_state]; 
                    numagents = numagents, 
                    maxsteps = maxsteps, 
                    inputmatrix = inputmatrix, 
                    noisematrix = noisematrix,
                    gating = true, 
                    seed = seed, 
                    kwargs...)
    wg_results = DataFrame(wg_data)
    return nog_results, wg_results, graph
end

function plot_sim(results; gating=false)
    CairoMakie.activate!() # hide
    if gating
        fig = Figure(size = (1100, 350))
    else
        fig = Figure(size = (1000, 400))
    end
    ax1 = fig[1, 1] = Axis(
            fig,
            xlabel = "Step",
            ylabel = "Opinion state",
            title = "Opinion formation",
            limits = (nothing, nothing, min(-0.5, minimum(results.opinion_new)), max(0.5, maximum(results.opinion_new)))
        )
    for grp in groupby(results, :id)
        lines!(ax1, grp.time*euler_h, grp.opinion_new, color = :blue, alpha = 0.4)
    end
    ax2 = fig[1, 2] = Axis(
            fig,
            xlabel = "Step",
            ylabel = "Susceptibility state",
            title = "Susceptibility over time",
            limits = (nothing, nothing, nothing, max(0.5, maximum(results.scp_state)))
        )
    for grp in groupby(results, :id)
        lines!(ax2, grp.time*euler_h, grp.scp_state, color = :red, alpha = 0.5)
    end
    if gating
        ax3 = fig[1, 3] = Axis(
                fig,
                xlabel = "Step",
                ylabel = "Gating state",
                title = "Gating state over time",
                limits = (nothing, nothing, min(-0.5, minimum(results.gating_state)), max(0.5, maximum(results.gating_state)))
            )
        for grp in groupby(results, :id)
            lines!(ax3, grp.time*euler_h, grp.gating_state, color = :green, alpha = 0.5)
        end
    end
    return fig
end

numagents = 100
euler_h = 0.01
maxsteps = Int(50 / euler_h)

myinputmatrix = create_extsignal(numagents, maxsteps,
                                prestimulus_delay = Int(5 / euler_h),
                                group1 = 0.25, signal1_nbbursts = 1, signal1_strength = "medium", 
                                signal1_dir = 1.0, signal1_timeon = Int(2 / euler_h),
                                group2 = 0.25, signal2_nbbursts = 1, signal2_strength = "medium", 
                                signal2_dir = -1.0, signal2_timeon = Int(2 / euler_h))
mynoisematrix = create_extnoise(numagents, maxsteps,
                                noise_mean = 0.0, noise_var = 0.0)

nog_res, wg_res, g = runsim_oneparam_g(numagents, maxsteps,
                    myinputmatrix, mynoisematrix, 
                    jz_network = ("wattsstrogatz", 5),
                    #jz_network = ("erdosrenyi", 5),
                    #jz_network = ("barabasialbert", 4),
                    jx_network = "buddies",
                    gain_scp=0.5,
                    seed=2854, 
                    dampingparam=(0.75, 0.0), 
                    scalingparam=(1, 0.25),
                    tau_gate=5.0,
                    alpha_gate=2.0, #beta_gate=0.0,
                    euler_h=euler_h)

nog_fig = plot_sim(nog_res, gating=false)
save("plots/nvar0_nog_homd_medsig_g05_WS5_s2854.png", nog_fig)

wg_fig = plot_sim(wg_res, gating=true)
save("plots/nvar0_wg_homd_medsig_g05_WS5_a2_taux5_s2854.png", wg_fig)

###
using GraphMakie
using NetworkLayout
function viz_socgraph(g, res)
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
    f, ax, p = graphplot(g; #layout=Spring(dim=2, seed=5),
                        node_marker=Circle, 
                        #nlabels=repr.(vertices(g)),
                        #nlabels_color=:black, nlabels_fontsize=11,
                        node_size=[sqrt(degree(g, i))*3 for i in vertices(g)], 
                        node_color=[sortedenddata.nodecolor[i] for i in vertices(g)], 
                        edge_color=:grey32, line_width=0.5)
    hidedecorations!(ax); hidespines!(ax)
    ax.aspect = DataAspect()
    return f, enddata, sortedenddata
end

f, _, _ = viz_socgraph(g, wg_res)
f

###
# From the results table, pick 2 agents and graph their trajectory in 3D state-space
# using the opinion, susceptibility, and gating states
# The 3D plot will show the trajectory of the agents in the state-space
# The trajectory will be colored by time, with the color gradient showing the progression of time
# The trajectory will be plotted in 3D, with the x-axis representing the opinion state, the y-axis representing the susceptibility state, 
# and the z-axis representing the gating state
# The trajectory will be plotted as a line, with the color of the line representing the progression of time

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

fig3d = plot_3dtrajectory(wg_res, 50)

# cmap = cgrad(:lightrainbow), color = cmap[grp.id[1]/100]
# agentcolor = cmap[grp.opinion_new[end] + 0.5]
###

model = init_bfl_model(gating = true)
Agents.step!(model, 20)
#model[1]

# generate draws from a normal mean 0, variance 0.1 distribution
#x = randn(1000) * 0.01

# Hyperparameters vectors
xdamping = [0.01, 0.1, 0.5, 1]
xbasal_scp = [0, 0.1, 0.5, 1]
xtau_scp = [0.01, 0.1, 0.5, 1]
xscp_gain = [0.01, 0.1, 0.5, 1]
xsaturation = ["tanh", "sigmoid"]
xbiastype = ["none", "manual", "randomsmall", "randomlarge", "time"]
xalpha = [1, 2, 3, 4, 5]
xbudnetworkparam = [("barabasialbert", xalpha[3]), ("erdosrenyi", xalpha[3]), ("wattsstrogatz", xalpha[3]), ("complete", xalpha[3])]
xattnetworkparam = ["identity", "buddies"]
xopinioninit = ["zero", "random", "binary"]
xbias = [0.1, 0.2, 0.3, 0.4, 0.5]

# Run and plot the model for different values of the hyper parameter
const cmap = cgrad(:lightrainbow)
plotsim(ax, data) =
    for grp in groupby(data, :id)
        lines!(ax, grp.time, grp.opinion_new, color = cmap[grp.id[1]/100])
    end

xs = [bias_process(xbias[i], 70, 80, maxsteps) for i in 1:4]
figure = Figure(size = (600, 1200))
for (i, x) in enumerate(xs)
    ax = figure[i, 1] = Axis(figure; title = "Bias strength index $i")
    x_data = model_run([:opinion_new]; 
        numagents = 13, 
        biastype = "manual",
        biasproc = x,
        opinioninit = "random",
        numsensing = 4,
        maxsteps = maxsteps,
        seed = 23)
    plotsim(ax, x_data)
end
figure
# Save to the plot subfolder
save("plots/barabasi_n13_k2_noatt_diffbiases.png", figure)
