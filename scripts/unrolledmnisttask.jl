using Random
using Distributions
using Flux
using Zygote
using CairoMakie
using LinearAlgebra
using ParameterSchedulers
using ParameterSchedulers: Scheduler
using MLDatasets

include("genrandommatrix.jl")
include("rnn_cells.jl")
include("customlossfunctions.jl")
include("plot_utils.jl")
include("train_utils.jl")
include("train_setup.jl")

#device = Flux.get_device(; verbose=true)

##### EXPERIMENTAL CONDITIONS

TASKCAT = "classification"
DATASETNAME = "MNIST"
MEASCAT = "identity"
TASK = "unrolled"

TURNS::Int = 1
VANILLA::Bool = true
GATED::Bool = false
net_width::Int = 1500

knownentries = 28*28

EPOCHS = 5
MINIBATCH_SIZE = 128

##########
# load MNIST
x_train, y_train = MNIST(split=:train)[:]
x_test,  y_test  = MNIST(split=:test)[:]
train_num, val_num = 50000, 10000
x_train = Flux.unsqueeze(x_train, 3)
x_train = reshape(x_train, 28*28, 60000)
Xval = x_train[:, 1:val_num]
Xtrain = x_train[:, (val_num+1):end]
Xtest = Flux.unsqueeze(x_test, 3)
Xtest = reshape(Xtest, 28*28, 10000)
# Encode labels
y_train = Float32.(Flux.onehotbatch(y_train, 0:9))
Yval = y_train[:, 1:val_num]
Ytrain = y_train[:, (val_num+1):end]
Ytest = Float32.(Flux.onehotbatch(y_test, 0:9))

m, n = 28, 28

size(Xtrain), size(Ytrain)

##### INITIALIZE NETWORK

Whh_init = nothing
input_size = 0
# Set output size based on task
output_size = 10

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
# Define methods
reset!(m::Chain) = reset!(m.layers[:rnn])
state(m::Chain) = state(m.layers[:rnn])

##### DEFINE LOSS FUNCTIONS
myloss = multiclass_classif_losses

##### TRAINING
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


# State optimizing rule
#=eta = 1e-3
opt = OptimiserChain(
    Adam(eta)
)=#
init_eta = 1e-3
decay = 0.8
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
train_acc = Float32[]
val_acc = Float32[]
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

initmetrics_train = myloss(activemodel, Xtrain[:, 1:1000], Ytrain[:, 1:1000]; 
                           turns = TURNS, include_accuracy = true)
initmetrics_val = myloss(activemodel, Xval, Yval; 
                         turns = TURNS, include_accuracy = true)
push!(train_loss, initmetrics_train["cross-entropy loss"])
push!(train_acc, initmetrics_train["accuracy"])
push!(val_loss, initmetrics_val["cross-entropy loss"])
push!(val_acc, initmetrics_val["accuracy"])

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
        train_loss_value, grads = Flux.withgradient(myloss, activemodel, x, y)#[1]
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
    trainmetrics = myloss(activemodel, Xtrain[:,1:1000], Ytrain[:,1:1000]; 
                          turns = TURNS, include_accuracy = true)
    push!(train_loss, trainmetrics["cross-entropy loss"])
    push!(train_acc, trainmetrics["accuracy"])
    # Compute validation metrics
    valmetrics = myloss(activemodel, Xval, Yval; 
                        turns = TURNS, include_accuracy = true)
    push!(val_loss, valmetrics["cross-entropy loss"])
    push!(val_acc, valmetrics["accuracy"])
    println("Epoch $epoch: Train loss: $(train_loss[end]); train acc: $(train_acc[end])")
    println("Epoch $epoch: Val loss: $(val_loss[end]); val acc: $(val_acc[end])")
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
testmetrics = myloss(activemodel, Xtest, Ytest; 
                     turns = TURNS, include_accuracy = true)
println("Test acc: $(testmetrics["accuracy"])")
println("Test loss: $(testmetrics["cross-entropy loss"])")


lossname = "Mean cross-entropy loss"
accname = "Accuracy"
tasklab = "Classifying unrolled MNIST with decentralized inputs"
taskfilename = "unrolledMNIST"

CairoMakie.activate!()
fig = Figure(size = (820, 450))
#epochs = length(train_loss) - 1
train_l = train_loss[126:end]
val_l = val_loss[2:end]
train_r = train_acc[2:end]
val_r = val_acc[2:end]
epochs = length(val_l)
ax_loss = Axis(fig[1, 1], xlabel = "Epochs", ylabel = "Loss", title = lossname)
# There are many samples of train loss (init + every mini batch) and few samples of val_loss (init + every epoch)
lines!(ax_loss, [i for i in range(1, epochs, length(train_l))], train_l, color = :blue, label = "Training")
lines!(ax_loss, 1:epochs, val_l, color = :red, label = "Validation")
lines!(ax_loss, 1:epochs, [testmetrics["cross-entropy"]], color = :green, linestyle = :dash)
scatter!(ax_loss, epochs, testmetrics["cross-entropy"], color = :green, label = "Final Test")
axislegend(ax_loss, backgroundcolor = :transparent)
ax_acc = Axis(fig[1, 2], xlabel = "Epochs", ylabel = "acc", title = accname)
#band!(ax_acc, 1:epochs, 0.995 .* ones(epochs), 1.005 .* ones(epochs), label = "Random guess", color = :gray, alpha = 0.25)
lines!(ax_acc, 1:epochs, train_r, color = :blue, label = "Training")
lines!(ax_acc, 1:epochs, val_r, color = :red, label = "Validation")
lines!(ax_acc, 1:epochs, [testmetrics["acc"]], color = :green, linestyle = :dash)
scatter!(ax_acc, epochs, testmetrics["acc"], color = :green, label = "Final Test")
axislegend(ax_acc, backgroundcolor = :transparent)
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
    "Test loss: $(round(testmetrics["cross-entropy"],digits=4))."*
    "Test acc: $(round(testmetrics["acc"],digits=4)).",
    #"Optimizer: AdamW(eta=$eta, beta=$beta, decay=$decay)",
    fontsize = 14,
    padding = (0, 0, 0, 0),
)
fig

save("data/$(modlabel)RNNwidth$(net_width)_$(taskfilename)_$(TURNS)turns_knownentries.png", fig)

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
