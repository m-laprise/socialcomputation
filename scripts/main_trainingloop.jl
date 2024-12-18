using Random
using Distributions
using Flux
using Zygote
using JLD2, CodecBzip2
using CairoMakie
using LinearAlgebra
using ParameterSchedulers
using ParameterSchedulers: Scheduler

include("genrandommatrix.jl")
include("rnn_cells.jl")
include("customlossfunctions.jl")
include("plot_utils.jl")
include("train_utils.jl")
include("train_setup.jl")

#device = Flux.get_device(; verbose=true)

##### EXPERIMENTAL CONDITIONS

taskcats = ["classification", 
            "reconstruction"]
measurecats = ["masks", 
               "traces", 
               "blocks"]
tasks = ["classif1a", "classif1b", 
         "classif2a", "classif2b", 
         "recon32", "recon64", "recon128", "recon256",
         "sparse200a", "sparse200b"]
datasetnames = ["lr_c1", "lr_c2", 
                "lr_r_32", "lr_r_64", "lr_r_128", "lr_r_256", 
                "sparse_200"]

TASKCAT = taskcats[2]
DATASETNAME = datasetnames[3]
MEASCAT = measurecats[1]
TASK = tasks[5]

RANK::Int = 1

TURNS::Int = 20
VANILLA::Bool = false
GATED::Bool = true
net_width::Int = 1500

if MEASCAT == "masks"
    knownentries = 1000
else 
    knownentries = nothing
end

EPOCHS = 6
MINIBATCH_SIZE = 64

INFERENCE_EXPERIMENT::Bool = false
WIDTH_EXPERIMENT::Bool = false
if WIDTH_EXPERIMENT
    widthvec = []
    sotamsevec = []
    sotaspectralvec = []
    trainlossvec = []
    trainaccvec = []
    testaccvec = []
    testlossvec = []
end
SOCGRAPHINIT::Bool = false

##########

train_prop::Float64 = 0.8
val_prop::Float64 = 0.1
test_prop::Float64 = 0.1

m, n, dataset_size = 80, 80, 10000
function setup(m, n, r, seed; datatype=Float32)
    rng = Random.MersenneTwister(seed)
    #v = randn(rng, datatype, m, r) ./ sqrt(sqrt(Float32(r)))
    #A = v * v'
    A = (randn(rng, datatype, m, r) ./ sqrt(sqrt(Float32(r)))) * (randn(rng, datatype, r, n) ./ sqrt(sqrt(Float32(r))))
    return A * 0.1f0
end
Y = Array{Float32, 3}(undef, m, n, dataset_size)
for i in 1:dataset_size
    Y[:, :, i] = setup(m, n, RANK, 111+i)
end
fixedmask = sensingmasks(m, n; k=knownentries, seed=9632)
mask_mat = masktuple2array(fixedmask)
@assert size(mask_mat) == (m, n)
X = input_setup(Y, MEASCAT, m, n, dataset_size, knownentries)

# Split data between training, validation, and test sets
Xtrain, Xval, Xtest = train_val_test_split(X, train_prop, val_prop, test_prop)
Ytrain, Yval, Ytest = train_val_test_split(Y, train_prop, val_prop, test_prop)

ranks = [RANK]

size(Xtrain), size(Ytrain)

##### INITIALIZE NETWORK

Whh_init = nothing
#Whh_init = load("data/Whh_init.jld2", "Whh_init")
if SOCGRAPHINIT
    include("inprogress/helpersWhh.jl")
    g = init_socgraph("Barabasi-Albert", net_width, 3, 9632)
    #g = init_socgraph("Erdos-Renyi", net_width, 3, 9632)
    #g = init_socgraph("Watts-Strogatz", net_width, 3, 9632)
    adj = Float32.(graph_to_adj(g))
    Whh_init = adj .+ 0.5*(randn(Float32, net_width, net_width) / sqrt(Float32(net_width)))
    #print_socgraph_descr(g)
    #plot_socgraph(g)
    #plot_degree_distrib(g)
end

# Set input size
input_size = 0

# Set output size based on task
# For matrix reconstruction, output size is the number of entries in the matrix
output_size = m * n

#Join(combine, paths) = Parallel(combine, paths)
#Join(combine, paths...) = Join(combine, paths)

# Initialize the vanilla RNN model
m_vanilla = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = false),
    dec = Dense(net_width => output_size),
)
# Initialize the GRU model
m_gru = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = true),
    filter = x -> x[:,1], # Use only opinion states, not gating state, for decoding
    dec = Dense(net_width => output_size),
)
# Initialize the BFL RNN model
m_bfl = Chain(
    rnn = rnn(input_size, net_width; 
        Whh_init = Whh_init, 
        h_init = "randn",
        bfl = true,
        basal_u = 0.01f0,
        gain = 1.25f0),
    filter = x -> x[:,1], # Use only opinion states, not attention states, for decoding
    dec = Dense(net_width => output_size)
)

##### DEFINE LOSS FUNCTIONS

myloss = recon_losses

##### TRAINING

#m_vanilla((Xtrain[:,1]))
#m_vanilla((Xtrain[:,2]))[1]
#c = myloss(m_vanilla, (Xtrain[:,3]), Ytrain[:,3], turns = turns)
#ab = myloss(m_vanilla, (Xtrain[:,2:3]), Ytrain[:,2:3], turns = turns)
#g = gradient(myloss, m_vanilla, (Xtrain[:,1:20]), Ytrain[:,1:20])[1]

if VANILLA
    activemodel = m_vanilla
    GATED = false
elseif GATED
    activemodel = m_gru
    GATED = true
else
    activemodel = m_bfl
    GATED = true
end
# Define method for the reset function
reset!(m::Chain) = reset!(m.layers[:rnn])
state(m::Chain) = state(m.layers[:rnn])

# State optimizing rule
#=eta = 1e-3
opt = OptimiserChain(
    Adam(eta)
)=#

#= min_eta = 1e-6 # don't actually start with lr = 0
initial_eta = 1e-2
warmup = 2 # warmup for 2 epochs
WarmupSin(starteta, initeta, warmup, total_iters, schedule) =
    Sequence(Sin(l0 = starteta, l1 = initeta, period = 2 * warmup) => warmup,
             schedule => total_iters)
s = WarmupSin(min_lr, initial_eta, warmup, total_iters, Exp(initial_eta, 0.5))=#
# schedule = Cos(λ0 = 1e-4, λ1 = 1e-2, period = 10)
init_eta = 1e-4
decay = 0.7
s = Exp(start = init_eta, decay = decay)
#l0 = 5e-4
#l1 = 1e-6
#s = CosAnneal(l0, l1, EPOCHS) #Int(round(EPOCHS/2))
for (eta, epoch) in zip(s, 1:EPOCHS)
    println("Epoch $epoch: learning rate = $eta")
end
opt = Adam()
# Tree of states
opt_state = Flux.setup(opt, activemodel)

traindataloader = Flux.DataLoader(
    (data=Xtrain, label=Ytrain), 
    batchsize=MINIBATCH_SIZE, 
    shuffle=true)

train_loss = Float32[]
val_loss = Float32[]
train_rmse = Float32[]
val_rmse = Float32[]
jacobian_spectra = []
Whh_spectra = []

# STORE INITIAL METRICS
reset!(activemodel)
if VANILLA 
    activemodel(randn(Float32, knownentries))
    hJ = state(activemodel)
else
    activemodel(randn(Float32, knownentries))
    hJ = state(activemodel)[:,1]
end
J = statejacobian(activemodel, hJ)
push!(jacobian_spectra, eigvals(J))
push!(Whh_spectra, eigvals(activemodel[:rnn].cell.Whh))

gt = false
initmetrics_train = myloss(activemodel, Xtrain[:, 1:1000], Ytrain[:, :, 1:1000], mask_mat, gt; 
                           turns = TURNS, mode = "testing", incltrainloss = true)
initmetrics_val = myloss(activemodel, Xval, Yval, mask_mat, gt; 
                         turns = TURNS, mode = "testing", incltrainloss = true)
push!(train_loss, initmetrics_train["l2"])
push!(train_rmse, initmetrics_train["RMSE"])
push!(val_loss, initmetrics_val["l2"])
push!(val_rmse, initmetrics_val["RMSE"])

starttime = time()
println("===================")
for (eta, epoch) in zip(s, 1:EPOCHS)
    reset!(activemodel)
    #println("Commencing epoch $epoch")
    println("Commencing epoch $epoch (eta = $(round(eta, digits=6)))")
    opt.eta = eta
    # Initialize counters for gradient diagnostics
    mb, n, v, e = 1, 0, 0, 0
    # Iterate over minibatches
    for (x, y) in traindataloader
        # Forward pass (to compute the loss) and backward pass (to compute the gradients)
        train_loss_value, grads = Flux.withgradient(myloss, activemodel, x, y, mask_mat, gt)#[1]
        # During training, use the backward pass to store the training loss after the previous epoch
        push!(train_loss, train_loss_value)
        # Diagnose the gradients
        _n, _v, _e = inspect_gradients(grads[1])
        n += _n
        v += _v
        e += _e
        # Use the optimizer and grads to update the trainable parameters; update the optimizer states
        Flux.update!(opt_state, activemodel, grads[1])
        if mb == 1 || mb % 5 == 0
            println("Minibatch $(epoch)--$(mb): loss of $(round(train_loss_value, digits=4))")
        end
        mb += 1
    end
    # Compute the Jacobian
    push!(Whh_spectra, eigvals(Flux.params(activemodel[:rnn].cell)[1]))
    reset!(activemodel)
    if VANILLA 
        activemodel(randn(Float32, knownentries))
        hJ = state(activemodel)
    else
        activemodel(randn(Float32, knownentries))
        hJ = state(activemodel)[:,1]
    end
    J = statejacobian(activemodel, hJ)
    try
        push!(jacobian_spectra, eigvals(J))
    catch err
        println("Jacobian computation failed after epoch $epoch")
        println("with error: $err")
    end
    # Print a summary of the gradient diagnostics for the epoch
    diagnose_gradients(n, v, e)
    # Compute training metrics -- this is a very expensive operation because it involves a forward pass over the entire training set
    # so I take a subset of the training set to compute the metrics
    trainmetrics = myloss(activemodel, Xtrain[:,1:1000], Ytrain[:,:,1:1000], mask_mat, gt; 
                          turns = TURNS, mode = "testing", incltrainloss = true)
    push!(train_loss, trainmetrics["l2"])
    push!(train_rmse, trainmetrics["RMSE"])
    # Compute validation metrics
    valmetrics = myloss(activemodel, Xval, Yval, mask_mat, gt; 
                        turns = TURNS, mode = "testing", incltrainloss = true)
    push!(val_loss, valmetrics["l2"])
    push!(val_rmse, valmetrics["RMSE"])
    println("Epoch $epoch: Train loss: $(train_loss[end]); train RMSE: $(train_rmse[end])")
    println("Epoch $epoch: Val loss: $(val_loss[end]); val RMSE: $(val_rmse[end])")
    # Check if validation loss has increased for 2 epochs in a row; if so, stop training
    if length(val_loss) > 2
        if val_loss[end] > val_loss[end-1] && val_loss[end-1] > val_loss[end-2]
            println("Early stopping at epoch $epoch")
            break
        end
    end
end
endtime = time()
# training time in minutes
println("Training time: $(round((endtime - starttime) / 60, digits=2)) minutes")
# Assess on testing set
testmetrics = myloss(activemodel, Xtest, Ytest, mask_mat, gt; 
                     turns = TURNS, mode = "testing", incltrainloss = true)
println("Test RMSE: $(testmetrics["RMSE"])")
println("Test loss: $(testmetrics["l2"])")


lossname = "Mean nuclear-norm penalized l2 loss (known entries)"
rmsename = "Root mean squared reconstruction error / std (all entries)"
tasklab = "Reconstructing $(m)x$(m) rank-$(RANK) matrices from $(knownentries) of their entries"
taskfilename = "$(m)recon"

CairoMakie.activate!()
fig = Figure(size = (820, 450))
#epochs = length(train_loss) - 1
train_l = train_loss[126:end]
val_l = val_loss[2:end]
train_r = train_rmse[2:end]
val_r = val_rmse[2:end]
epochs = length(val_l)
ax_loss = Axis(fig[1, 1], xlabel = "Epochs", ylabel = "Loss", title = lossname)
# There are many samples of train loss (init + every mini batch) and few samples of val_loss (init + every epoch)
lines!(ax_loss, [i for i in range(1, epochs, length(train_l))], train_l, color = :blue, label = "Training")
lines!(ax_loss, 1:epochs, val_l, color = :red, label = "Validation")
lines!(ax_loss, 1:epochs, [testmetrics["l2"]], color = :green, linestyle = :dash)
scatter!(ax_loss, epochs, testmetrics["l2"], color = :green, label = "Final Test")
axislegend(ax_loss, backgroundcolor = :transparent)
ax_rmse = Axis(fig[1, 2], xlabel = "Epochs", ylabel = "RMSE", title = rmsename)
#band!(ax_rmse, 1:epochs, 0.995 .* ones(epochs), 1.005 .* ones(epochs), label = "Random guess", color = :gray, alpha = 0.25)
lines!(ax_rmse, 1:epochs, train_r, color = :blue, label = "Training")
lines!(ax_rmse, 1:epochs, val_r, color = :red, label = "Validation")
lines!(ax_rmse, 1:epochs, [testmetrics["RMSE"]], color = :green, linestyle = :dash)
scatter!(ax_rmse, epochs, testmetrics["RMSE"], color = :green, label = "Final Test")
axislegend(ax_rmse, backgroundcolor = :transparent)
modlabel = GATED ? "Gated" : "Vanilla"
Label(
    fig[begin-1, 1:2],
    "$(tasklab)\n$(modlabel) RNN of $net_width units, $TURNS dynamic steps"*
    "\nwith training loss based on known entries",
    fontsize = 20,
    padding = (0, 0, 0, 0),
)
# Add notes to the bottom of the figure
Label(
    fig[end+1, 1:2],
    "Optimizer: Adam with schedule Exp(start = $(init_eta), decay = $(decay))\n"*
    #"Training time: $(round((endtime - starttime) / 60, digits=2)) minutes; "*
    "Test loss: $(round(testmetrics["l2"],digits=4))."*
    "Test RMSE: $(round(testmetrics["RMSE"],digits=4)).",
    #"Optimizer: AdamW(eta=$eta, beta=$beta, decay=$decay)",
    fontsize = 14,
    padding = (0, 0, 0, 0),
)
fig

save("data/$(modlabel)RNNwidth$(net_width)_$(taskfilename)_$(TURNS)turns_knownentries.png", fig)

if WIDTH_EXPERIMENT
    push!(widthvec, net_width)
    push!(sotamsevec, sota_test_mse)
    push!(sotaspectralvec, sota_test_spectraldist)
    push!(trainlossvec, train_loss[end])
    push!(testlossvec, test_loss)
    push!(trainaccvec, train_accuracy[end])
    push!(testaccvec, test_accuracy)
    
    fig_w = Figure(size = (850, 380))
    ax1 = Axis(fig_w[1, 1], xlabel = "Width of RNN and nb of entries", ylabel = "Mean squared reconstruction error", 
               title = "End of training error")
    lines!(ax1, Int.(widthvec), Float64.(sotamsevec), color = :blue, linestyle = :dash, label = "Scaled ASD")
    lines!(ax1, Int.(widthvec), Float64.(trainlossvec), color = :red, label = "RNN on training data")
    lines!(ax1, Int.(widthvec), Float64.(testlossvec), color = :green, label = "RNN on test data")
    axislegend(ax1, backgroundcolor = :transparent)
    ax2 = Axis(fig_w[1, 2], xlabel = "Width of RNN and nb of entries", ylabel = "Mean norm of spectral distance", 
               title = "End of training spectral distance")
    lines!(ax2, Int.(widthvec), Float64.(sotaspectralvec), color = :blue, linestyle = :dash, label = "Scaled ASD")
    lines!(ax2, Int.(widthvec), Float64.(trainaccvec), color = :red, label = "RNN on training data")
    lines!(ax2, Int.(widthvec), Float64.(testaccvec), color = :green, label = "RNN on test data")
    axislegend(ax2, backgroundcolor = :transparent)
    Label(
        fig_w[begin-1, 1:2],
        "Reconstructing 32x32 rank-8 matrices from K of their entries,\nwith $(modlabel) RNNs of width K, for K between 10 and 100",
        fontsize = 20,
        padding = (0, 0, 0, 0),
    )
    fig_w
    save("data/$(modlabel)RNNvaryingwidths_$(taskfilename)_$(TURNS)turns.png", fig_w)
end

Whh = activemodel[:rnn].cell.Whh
bs = activemodel[:rnn].cell.b
if GATED
    #gains = activemodel[:rnn].cell.gain
    #tauh = activemodel[:rnn].cell.tauh
    Wz = activemodel[:rnn].cell.Wz
    bz = activemodel[:rnn].cell.bz
end
if GATED
    height = 550
else
    height = 400
end

fig2 = Figure(size = (700, height))
ax1 = Axis(fig2[1, 1], title = "Eigenvalues of Recurrent Weights", xlabel = "Real", ylabel = "Imaginary")
θ = LinRange(0, 2π, 1000)
scatter!(ax1, real(eigvals(Whh)), imag(eigvals(Whh)), color = :blue, alpha = 0.85, markersize = 5)
lines!(ax1, cos.(θ), sin.(θ), color = :black, linewidth = 2)
ax2 = Axis(fig2[1, 2], title = "Recurrent Biases", xlabel = "Unit", ylabel = "Value")
scatter!(ax2, 1:net_width, bs, color = :red, alpha = 0.85, markersize = 5)
if GATED
    ax3 = Axis(fig2[2, 1], title = "Eigenvalues of Gating weights", xlabel = "Unit", ylabel = "Value")
    #scatter!(ax3, 1:net_width, Wz, color = :green, alpha = 0.85, markersize = 6)
    scatter!(ax3, real(eigvals(Wz)), imag(eigvals(Wz)), color = :blue, alpha = 0.85, markersize = 5)
    lines!(ax3, cos.(θ), sin.(θ), color = :black, linewidth = 2)
    ax4 = Axis(fig2[2, 2], title = "Gating biases", xlabel = "Unit", ylabel = "Value")
    scatter!(ax4, 1:net_width, bz, color = :purple, alpha = 0.85, markersize = 5)
end
fig2

#save("data/$(modlabel)RNNwidth$(net_width)_$(taskfilename)_$(TURNS)turns_Whh.jld2", "Whh", Whh)
save("data/$(modlabel)RNNwidth$(net_width)_$(taskfilename)_$(TURNS)turns_knownentries_learnedparams.png", fig2)

include("inprogress/helpersWhh.jl")
g_end = adj_to_graph(Whh; threshold = 0.01)
print_socgraph_descr(g_end)
figdegdist = plot_degree_distrib(g_end)
save("data/$(modlabel)RNNwidth$(net_width)_$(taskfilename)_$(TURNS)turns_knownentries_degreedist.png", figdegdist)
plot_socgraph(g_end)

if INFERENCE_EXPERIMENT
    k = 50
    forwardpasses = [i for i in 1:k]
    mse_by_nbpasses = zeros(Float32, length(forwardpasses))
    spectraldist_by_nbpasses = zeros(Float32, length(forwardpasses))
    for (i, nbpasses) in enumerate(forwardpasses)
        mse_by_nbpasses[i] = myloss(activemodel, Xtest, Ytest, turns = nbpasses)
        spectraldist_by_nbpasses[i] = accuracy(activemodel, Xtest, Ytest, turns = nbpasses)
    end

    fig3 = Figure(size = (700, 400))
    ax1 = Axis(fig3[1, 1], title = "", 
            xlabel = "Forward passes at inference time", ylabel = "Reconstruction MSE")
    lines!(ax1, forwardpasses, (mse_by_nbpasses), color = :blue, label = "RNN")
    scatter!(ax1, [TURNS], [test_loss], color = :blue, marker = :diamond, markersize = 12, 
            label = "EOT test: $(round(test_loss,digits=3))")
    min_mse = minimum(mse_by_nbpasses)
    min_mse_idx = argmin(mse_by_nbpasses)
    scatter!(ax1, [min_mse_idx], [min_mse], color = :blue, markersize = 12, 
            label = "Min. achieved: $(round(min_mse,digits=3))")
    hlines!(ax1, ([sota_test_mse]), color = :blue, linestyle = :dash, label = "Scaled ASD: $(round(sota_test_mse,digits=3))")
    ax2 = Axis(fig3[1, 2], title = "", 
            xlabel = "Forward passes at inference time", ylabel = "Spectral distance")
    lines!(ax2, forwardpasses, (spectraldist_by_nbpasses), color = :red, label = "RNN")
    min_spectraldist = minimum(spectraldist_by_nbpasses)
    min_spectraldist_idx = argmin(spectraldist_by_nbpasses)
    scatter!(ax2, [TURNS], [test_accuracy], color = :red, marker = :diamond, markersize = 12, 
            label = "EOT test: $(round(test_accuracy,digits=4))")
    scatter!(ax2, [min_spectraldist_idx], [min_spectraldist], color = :red, markersize = 12, 
            label = "Min. achieved: $(round(min_spectraldist,digits=4))")
    #ylims!(ax1, 37.0, maximum(mse_by_nbpasses) + 1.0)
    #ylims!(ax2, 4.5, maximum(spectraldist_by_nbpasses) + 0.1)
    axislegend(ax1, backgroundcolor = :transparent)
    axislegend(ax2, backgroundcolor = :transparent, position = :rb)
    Label(fig3[begin-1, 1:2], 
        "$(tasklab)\n$(modlabel) RNN of $net_width units, trained with $TURNS forward passes\nRecurrent inference on test set", 
        fontsize = 16)
    Label(fig3[end+1, 1:2], 
        "EOT stands for end of training. Spectral distance comparison is not shown because\n"*
        "the Scaled ASD algorithm uses the true ranks of the underlying ground truth, while\n"*
        "the RNN has no rank information.", fontsize = 12)
    fig3

    save("data/$(modlabel)RNNwidth$(net_width)_$(taskfilename)_$(TURNS)turns_inferencetests.png", fig3)
end

ploteigvals(jacobian_spectra[1])
ploteigvals(jacobian_spectra[end])

ploteigvals(Whh_spectra[1])
ploteigvals(Whh_spectra[end])

# Create animation using each ploteigvals(jacobian_spectra[i]) as one frame
using GLMakie
GLMakie.activate!()

N = length(jacobian_spectra)
ftime = Observable(1.0)
framerate = 2
timestamps = 1:N

# Find maximum value of the spectrum to set the limits of the plot
maxval = maximum([maximum(abs.(eigsvec)) for eigsvec in jacobian_spectra]) + 0.1
maxval = min(maxval, 5.0)

fj = Figure(size = (700, 700))
ax = Axis(fj[1, 1], xlabel = "Real", ylabel = "Imaginary",
          title = @lift("Spectrum of state-to-state Jacobian during training\n"*
                        "$(tasklab)\n$(modlabel) RNN of $net_width units, trained with $TURNS forward passes (known entries loss)\n"*
                        "Epoch = $(round($ftime[], digits = 0))"))
θ = LinRange(0, 2π, 1000)
lines!(ax, cos.(θ), sin.(θ), color = :black)
xs = @lift(real(jacobian_spectra[Int(round($ftime[]))]))
ys = @lift(imag(jacobian_spectra[Int(round($ftime[]))]))
scatter!(ax, xs, ys, color = :blue, alpha = 0.95, markersize = 6.5)
xlims!(ax, -maxval, maxval)
ylims!(ax, -maxval, maxval)

record(fj, "data/$(modlabel)RNNwidth$(net_width)_$(taskfilename)_$(TURNS)turns_knownentries_Jacobian.mp4", timestamps;
       framerate = framerate) do t
    ftime[] = t
    #autolimits!(ax)
end

N = length(Whh_spectra)
ftime = Observable(1.0)
framerate = 2
timestamps = 1:N

# Find maximum value of the spectrum to set the limits of the plot
maxval = maximum([maximum(abs.(eigsvec)) for eigsvec in Whh_spectra]) + 0.1
maxval = min(maxval, 5.0)

fw = Figure(size = (700, 600))
ax = Axis(fw[1, 1], xlabel = "Real", ylabel = "Imaginary",
          title = @lift("Spectrum of Whh weight matrix during training\n"*
                        "$(tasklab)\n$(modlabel) RNN of $net_width units, trained with $TURNS forward passes (known entries loss)\n"*
                        "Epoch = $(round($ftime[], digits = 0))"))
θ = LinRange(0, 2π, 1000)
lines!(ax, cos.(θ), sin.(θ), color = :black)
xs = @lift(real(Whh_spectra[Int(round($ftime[]))]))
ys = @lift(imag(Whh_spectra[Int(round($ftime[]))]))
scatter!(ax, xs, ys, color = :blue, alpha = 0.95, markersize = 6.5)
xlims!(ax, -maxval, maxval)
ylims!(ax, -maxval, maxval)
fw
record(fw, "data/$(modlabel)RNNwidth$(net_width)_$(taskfilename)_$(TURNS)turns_knownentries_Whh.mp4", timestamps;
       framerate = framerate) do t
    ftime[] = t
    #autolimits!(ax)
end
