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
    data = model_run([:opinion_new, :scp_state, :gating_state]; 
                     numagents = numagents, 
                     maxsteps = maxsteps, 
                     inputmatrix = inputmatrix, 
                     noisematrix = noisematrix,
                     gating = gating, 
                     seed = seed, 
                     kwargs...)
    results = DataFrame(data)
    return results
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
maxsteps = Int(10 / euler_h)

myinputmatrix = create_extsignal(numagents, maxsteps,
                                prestimulus_delay = Int(5 / euler_h),
                                group1 = 0.25, signal1_nbbursts = 1, signal1_strength = "medium", 
                                signal1_dir = 1.0, signal1_timeon = Int(2 / euler_h),
                                group2 = 0.25, signal2_nbbursts = 1, signal2_strength = "medium", 
                                signal2_dir = -1.0, signal2_timeon = Int(2 / euler_h))
mynoisematrix = create_extnoise(numagents, maxsteps,
                                noise_mean = 0.0, noise_var = 0.05)

res = runsim_oneparam(numagents, maxsteps,
                    myinputmatrix, mynoisematrix, 
                    gating=false, 
                    seed=25, 
                    dampingparam=(0.75, 0.25), 
                    scalingparam=(1, 0.25),
                    #alpha_gate=0.0, beta_gate=4.0,
                    euler_h=euler_h)
fig = plot_sim(res, gating=true)
fig

save("plots/wg_hetd_medsig.png", fig)

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
