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
maxsteps = Int(100 / euler_h)
seed = 2854
signal = "medium"
grapht = "er"
gparam = 5
myinputmatrix = create_extsignal(numagents, maxsteps,
                                prestimulus_delay = Int(5 / euler_h),
                                group1 = 0.25, signal1_nbbursts = 1, signal1_strength = signal, 
                                signal1_dir = 1.0, signal1_timeon = Int(2 / euler_h),
                                group2 = 0.25, signal2_nbbursts = 1, signal2_strength = signal, 
                                signal2_dir = -1.0, signal2_timeon = Int(2 / euler_h))
mynoisematrix = create_extnoise(numagents, maxsteps,
                                noise_mean = 0.0, noise_var = 0.0)

nog_res, wg_res, g = runsim_oneparam_g(numagents, maxsteps,
                    myinputmatrix, mynoisematrix, 
                    jz_network = (grapht, gparam),
                    jx_network = "buddies",
                    gain_scp=0.5,
                    seed=seed, 
                    dampingparam=(0.75, 0.0), 
                    scalingparam=(1, 0.25),
                    tau_gate=5.0,
                    alpha_gate=2.0, #beta_gate=0.0,
                    euler_h=euler_h)
nog_fig = plot_sim(nog_res, gating=false)
#save("plots/nvar0_nog_homd_$(signal)sig_g05_$(grapht)$(gparam)_s$(seed)_$(maxsteps).png", nog_fig)
wg_fig = plot_sim(wg_res, gating=true)
#save("plots/nvar0_wg_homd_$(signal)sig_g05_$(grapht)$(gparam)_a2_taux5_s$(seed)_$(maxsteps).png", wg_fig)

include("plot_3d_trajectories.jl")
include("plot_steadystategraph.jl")

using WGLMakie
WGLMakie.activate!()

fig3d = plot_3dtrajectory(wg_res, 50)

fg, _, _ = viz_socgraph(g, wg_res, 3)
fg


#model, g = init_bfl_model(gating = true)
#Agents.step!(model, 20)
#model[1]

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
#save("plots/barabasi_n13_k2_noatt_diffbiases.png", figure)
