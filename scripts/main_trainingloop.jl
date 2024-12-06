using Random
using Distributions
using Flux
using Zygote
using JLD2, CodecBzip2
using CairoMakie
using LinearAlgebra

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

DATASETNAME = datasetnames[3]
TASKCAT = taskcats[2]
MEASCAT = measurecats[1]
TASK = tasks[5]

TURNS::Int = 10
VANILLA::Bool = true
net_width::Int = 800

if MEASCAT == "masks"
    knownentries = 750
else 
    knownentries = nothing
end

EPOCHS = 50
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

#= Initial setup from saved data
data = load("data/rnn_firstexpdata.jld2", DATASETNAME)
m, n, dataset_size = size(data["X"])
# Create label data
Y = label_setup(data, TASKCAT, TASK)
# Create input data from ground truth
fixedmask = sensingmasks(m, n; k=knownentries, seed=9632)
X = input_setup(Y, MEASCAT, m, n, dataset_size, knownentries)
=#

m, n, dataset_size = 80, 80, 10000
function setup(m, n, r, seed; datatype=Float32)
    rng = Random.MersenneTwister(seed)
    A = (randn(rng, datatype, m, r) ./ sqrt(sqrt(Float32(r)))) * (randn(rng, datatype, r, n) ./ sqrt(sqrt(Float32(r))))
    return A
end
Y = Array{Float32, 3}(undef, m, n, dataset_size)
for i in 1:dataset_size
    Y[:, :, i] = setup(m, n, 1, 111+i)
end
fixedmask = sensingmasks(m, n; k=knownentries, seed=9632)
mask_mat = masktuple2array(fixedmask)
@assert size(mask_mat) == (m, n)
X = input_setup(Y, MEASCAT, m, n, dataset_size, knownentries)

# Split data between training, validation, and test sets
Xtrain, Xval, Xtest = train_val_test_split(X, train_prop, val_prop, test_prop)
Ytrain, Yval, Ytest = train_val_test_split(Y, train_prop, val_prop, test_prop)

ranks = [1]
#= if TASKCAT == "reconstruction"
    ranks = unique(data["ranks"])
    size(ranks)
else
    ranks_train, ranks_val, ranks_test = train_val_test_split(data["ranks"], train_prop, val_prop, test_prop)
    size(ranks_train)
end =#
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
if TASKCAT == "classification"
    # For binary classification, output size is 1
    if TASK == "classif1a" || TASK == "classif1b"
        output_size = 1
    # For multi-class classification, output size is the number of classes
    elseif TASK == "classif2a" || TASK == "classif2b"
        output_size = nb_classes
    end
elseif TASKCAT == "reconstruction"
    # For matrix reconstruction, output size is the number of entries in the matrix
    output_size = m * n
end

# Initialize the models for classification with a sigmoid 
# and the models for reconstruction without a sigmoid
#Join(combine, paths) = Parallel(combine, paths)
#Join(combine, paths...) = Join(combine, paths)

if TASKCAT == "classification"
    # Initialize the vanilla RNN model
    m_vanilla = Chain(
        rnn = rnn(input_size, net_width; 
            Whh_init = Whh_init, 
            h_init = "randn",
            gated = false),
        dec = Dense(net_width => output_size, sigmoid)
    )
    # Initialize the BFL RNN model
    m_bfl = Chain(
        rnn = rnn(input_size, net_width;
            Whh_init = Whh_init,
            h_init = "randn",
            gated = true,
            basal_u = 0.01f0,
            gain = 0.5f0),
        filter = x -> x[:,1], # Use only opinion states, not attention states, for decoding
        dec = Dense(net_width => output_size, sigmoid)
    )
elseif TASKCAT == "reconstruction"
    # Initialize the vanilla RNN model
    m_vanilla = Chain(
        rnn = rnn(input_size, net_width;
            Whh_init = Whh_init, 
            h_init = "randn",
            gated = false, dual = true),
        combine = x -> (x[:,1] * x[:,2]'),
        dec = BasisChange(net_width => m),
        flatten = x -> vec(x)
    )
    # Initialize the BFL RNN model
    m_bfl = Chain(
        rnn = rnn(input_size, net_width; 
            Whh_init = Whh_init, 
            h_init = "randn",
            gated = true,
            basal_u = 0.01f0,
            gain = 1.25f0),
        filter = x -> x[:,1], # Use only opinion states, not attention states, for decoding
        dec = Dense(net_width => output_size)
    )
end

##### DEFINE LOSS FUNCTIONS

if TASKCAT == "classification" 
    myloss = logitbinarycrossent
    accuracy = classification_accuracy
elseif TASKCAT == "reconstruction"
    myloss = recon_losses
end

##### TRAINING

#m_vanilla((Xtrain[:,1]))
#m_vanilla((Xtrain[:,2]))[1]
#c = myloss(m_vanilla, (Xtrain[:,3]), Ytrain[:,3], turns = turns)
#ab = myloss(m_vanilla, (Xtrain[:,2:3]), Ytrain[:,2:3], turns = turns)
#g = gradient(myloss, m_vanilla, (Xtrain[:,1:20]), Ytrain[:,1:20])[1]

if VANILLA
    activemodel = m_vanilla
    GATED = false
else
    activemodel = m_bfl
    GATED = true
end
# Define method for the reset function
reset!(m::Chain) = reset!(m.layers[:rnn])
state(m::Chain) = state(m.layers[:rnn])

# State optimizing rule
eta = 1e-3
#beta, decay = (0.89, 0.995), 0.1
#omega = 10
opt = OptimiserChain(
    #ClipNorm(omega), AdamW(eta, beta, decay)
    Adam(eta)
)
# Tree of states
opt_state = Flux.setup(opt, activemodel)

traindataloader = Flux.DataLoader(
    (data=Xtrain, label=Ytrain), 
    batchsize=MINIBATCH_SIZE, 
    shuffle=true)

train_loss = Float32[]
val_loss = Float32[]
#train_accuracy = Float32[]
#val_accuracy = Float32[]
train_rmse = Float32[]
val_rmse = Float32[]
jacobian_spectra = []
jacobian_spectra2 = []
Whh_spectra = []

# STORE INITIAL METRICS
#=reset!(activemodel)
if VANILLA 
    hJ = state(activemodel)
else
    activemodel(randn(Float32, knownentries))
    hJ = state(activemodel)[:,1]
end
J = statejacobian(activemodel, hJ)=#
reset!(activemodel)
activemodel(randn(Float32, knownentries))
hJ = state(activemodel)[:,1]
J = statejacobian(activemodel, hJ)
reset!(activemodel)
activemodel(randn(Float32, knownentries))
hJ2 = state(activemodel)[:,2]
J2 = statejacobian(activemodel, hJ2)
try
    push!(jacobian_spectra, eigvals(J))
    push!(jacobian_spectra2, eigvals(J2))
catch err
    println("Jacobian spectrum computation failed after epoch $epoch")
    println("with error: $err")
end
push!(Whh_spectra, eigvals(activemodel[:rnn].cell.Whh))

gt = false
initmetrics_train = myloss(activemodel, Xtrain[:, 1:1000], Ytrain[:, :, 1:1000], mask_mat, gt; 
                           turns = TURNS, mode = "testing", incltrainloss = true)
initmetrics_val = myloss(activemodel, Xval, Yval, mask_mat, gt; 
                         turns = TURNS, mode = "testing", incltrainloss = true)
push!(train_loss, initmetrics_train["l2"])
#push!(train_accuracy, initmetrics_train["spectdist"])
push!(train_rmse, initmetrics_train["RMSE"])
push!(val_loss, initmetrics_val["l2"])
#push!(val_accuracy, initmetrics_val["spectdist"])
push!(val_rmse, initmetrics_val["RMSE"])

starttime = time()
for epoch in 1:5
    reset!(activemodel)
    println("Commencing epoch $epoch")
    # Initialize counters for gradient diagnostics
    mb, n, v, e = 1, 0, 0, 0
    # Iterate over minibatches
    for (x, y) in traindataloader
        # Forward pass (to compute the loss) and backward pass (to compute the gradients)
        train_loss_value, grads = Flux.withgradient(myloss, activemodel, x, y, mask_mat, gt)#[1]
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
    #=reset!(activemodel)
    if VANILLA 
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
    end=#
    # Print a summary of the gradient diagnostics for the epoch
    diagnose_gradients(n, v, e)
    # Compute training metrics -- this is a very expensive operation because it involves a forward pass over the entire training set
    # so I take a subset of the training set to compute the metrics
    trainmetrics = myloss(activemodel, Xtrain[:,1:1000], Ytrain[:,:,1:1000], mask_mat, gt; 
                          turns = TURNS, mode = "testing", incltrainloss = false)
    #push!(train_accuracy, trainmetrics["spectdist"])
    push!(train_rmse, trainmetrics["RMSE"])
    # Compute validation metrics
    valmetrics = myloss(activemodel, Xval, Yval, mask_mat, gt; 
                        turns = TURNS, mode = "testing", incltrainloss = true)
    push!(val_loss, valmetrics["l2"])
    #push!(val_accuracy, valmetrics["spectdist"])
    push!(val_rmse, valmetrics["RMSE"])
    println("Epoch $epoch: Train loss: $(train_loss[end]); train RMSE: $(train_rmse[end])")
end
endtime = time()
# training time in minutes
println("Training time: $(round((endtime - starttime) / 60, digits=2)) minutes")
# Compute test accuracy
testmetrics = myloss(activemodel, Xtest, Ytest, mask_mat, gt; 
                     turns = TURNS, mode = "testing", incltrainloss = true)
#println("Test spectdist: $(testmetrics["spectdist"])")
println("Test RMSE: $(testmetrics["RMSE"])")
println("Test loss: $(testmetrics["l2"])")


if TASKCAT == "reconstruction"
    lossname = "Mean nuclear-norm penalized l2 loss (known entries)"
    #accuracyname = "Mean norm of spectral distance (singular values)"
    rmsename = "Root mean squared reconstruction error (all entries)"
else
    lossname = "Binary cross-entropy loss"
    accuracyname = "Classification accuracy"
end

tasklab = "Reconstructing 80x80 rank-1 matrices from $(knownentries) of their entries"
taskfilename = "80recon"

#if TASK == "recon32"
#    tasklab = "Reconstructing 32x32 rank 8 matrices from $(net_width) of their entries"
#    taskfilename = "32recon"
#= elseif TASK == "small_classification"
    tasklab = "Classifying 8x8 matrices as full rank or rank 1"
    taskfilename = "8by8class"
elseif TASK == "random traces"
    tasklab = "Classifying 250x250 matrices as low/high rank from 50 traces of random projections"
    taskfilename = "250by250tracesclass" =#
#end
CairoMakie.activate!()
fig = Figure(size = (820, 450))
#fig = Figure(size = (820, 700))
#epochs = length(train_loss) - 1
epochs = length(val_loss)-1
ax_loss = Axis(fig[1, 1], xlabel = "Epochs", ylabel = "Loss", title = lossname)
# There are 626 samples of train loss (init + every mini batch) and only 6 samples of val_loss (init + every epoch)
lines!(ax_loss, [i for i in range(1, epochs+1, length(train_loss))], train_loss, color = :blue, label = "Training")
#lines!(ax_loss, 1:epochs+1, train_loss, color = :blue, label = "Training")
lines!(ax_loss, 1:epochs+1, val_loss, color = :red, label = "Validation")
lines!(ax_loss, 1:epochs+1, [testmetrics["l2"]], color = :green, linestyle = :dash, label = "Final Test")
#ylims!(ax_loss, 35.0, 170.0)
axislegend(ax_loss, backgroundcolor = :transparent)
#= ax_acc = Axis(fig[1, 2], xlabel = "Epochs", ylabel = "", title = accuracyname)
lines!(ax_acc, 1:epochs+1, train_accuracy, color = :blue, label = "Training")
lines!(ax_acc, 1:epochs+1, val_accuracy, color = :red, label = "Validation")
hlines!(ax_acc, [testmetrics["spectdist"]], color = :green, linestyle = :dash, label = "Final Test") 
#ylims!(ax_acc, 4.0, 7.0)
axislegend(ax_acc, backgroundcolor = :transparent, position = :rt)=#
ax_rmse = Axis(fig[1, 2], xlabel = "Epochs", ylabel = "RMSE", title = rmsename)
band!(ax_rmse, 1:epochs+1, 0.995 .* ones(epochs+1), 1.005 .* ones(epochs+1), label = "Random guess", color = :gray, alpha = 0.25)
lines!(ax_rmse, 1:epochs+1, train_rmse, color = :blue, label = "Training")
lines!(ax_rmse, 1:epochs+1, val_rmse, color = :red, label = "Validation")
lines!(ax_rmse, 1:epochs+1, [testmetrics["RMSE"]], color = :green, linestyle = :dash, label = "Final Test")
axislegend(ax_rmse, backgroundcolor = :transparent)
# Add a title to the top of the figure
modlabel = GATED ? "BFL" : "Vanilla"
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
    "Optimizer: Adam(eta=0.002 for 5 epochs, then eta=$eta)\n"*
    #"Training time: $(round((endtime - starttime) / 60, digits=2)) minutes; "*
    "Test loss: $(round(testmetrics["l2"],digits=2))."*
    #"Test spectral distance: $(round(testmetrics["spectdist"],digits=2))."*
    "Test RMSE: $(round(testmetrics["RMSE"],digits=2)).",
    #"Optimizer: AdamW(eta=$eta, beta=$beta, decay=$decay)\nTraining time: $(round((endtime - starttime) / 60, digits=2)) minutes; Test accuracy: $(round(test_accuracy,digits=2)).",
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
    gains = activemodel[:rnn].cell.gain
    tauh = activemodel[:rnn].cell.tauh
end
if GATED
    height = 550
else
    height = 400
end


fig2 = Figure(size = (700, height))
ax1 = Axis(fig2[1, 1], title = "Eigenvalues of Recurrent Weights", xlabel = "Real", ylabel = "Imaginary")
#ploteigvals!(ax1, Whh; alpha = 0.5)
θ = LinRange(0, 2π, 1000)
scatter!(ax1, real(eigvals(Whh)), imag(eigvals(Whh)), color = :blue, alpha = 0.85, markersize = 6)
lines!(ax1, cos.(θ), sin.(θ), color = :black, linewidth = 2)
ax2 = Axis(fig2[1, 2], title = "Biases", xlabel = "Unit", ylabel = "Value")
scatter!(ax2, 1:net_width, bs, color = :red, alpha = 0.85, markersize = 6)
if GATED
    ax3 = Axis(fig2[2, 1], title = "Tau", xlabel = "Unit", ylabel = "Value")
    scatter!(ax3, 1:net_width, tauh, color = :green, alpha = 0.85, markersize = 6)
    ax4 = Axis(fig2[2, 2], title = "Gains", xlabel = "Unit", ylabel = "Value")
    scatter!(ax4, 1:net_width, gains, color = :purple, alpha = 0.85, markersize = 6)
end
fig2

save("data/$(modlabel)RNNwidth$(net_width)_$(taskfilename)_$(TURNS)turns_Whh.jld2", "Whh", Whh)
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

fj = Figure()
ax = Axis(fj[1, 1], 
          title = @lift("Spectrum of state-to-state Jacobian during training\n"*
                        "$(tasklab)\n$(modlabel) RNN of $net_width units, trained with $TURNS forward passes (known entries loss)\n"*
                        "Epoch = $(round($ftime[], digits = 0))"))
θ = LinRange(0, 2π, 1000)
lines!(ax, cos.(θ), sin.(θ), color = :black)
xs = @lift(real(jacobian_spectra[Int(round($ftime[]))]))
ys = @lift(imag(jacobian_spectra[Int(round($ftime[]))]))
scatter!(ax, xs, ys, color = :blue, alpha = 0.95, markersize = 6.5)

record(fj, "data/$(modlabel)RNNwidth$(net_width)_$(taskfilename)_$(TURNS)turns_knownentries_Jacobian.mp4", timestamps;
       framerate = framerate) do t
    ftime[] = t
end


N = length(Whh_spectra)
ftime = Observable(1.0)
framerate = 2
timestamps = 1:N

fw = Figure()
ax = Axis(fw[1, 1], 
          title = @lift("Spectrum of Whh weight matrix during training\n"*
                        "$(tasklab)\n$(modlabel) RNN of $net_width units, trained with $TURNS forward passes (known entries loss)\n"*
                        "Epoch = $(round($ftime[], digits = 0))"))
θ = LinRange(0, 2π, 1000)
lines!(ax, cos.(θ), sin.(θ), color = :black)
xs = @lift(real(Whh_spectra[Int(round($ftime[]))]))
ys = @lift(imag(Whh_spectra[Int(round($ftime[]))]))
scatter!(ax, xs, ys, color = :blue, alpha = 0.95, markersize = 6.5)
xlims!(ax, -1.5, 1.5)
ylims!(ax, -1.5, 1.5)

record(fw, "data/$(modlabel)RNNwidth$(net_width)_$(taskfilename)_$(TURNS)turns_knownentries_Whh.mp4", timestamps;
       framerate = framerate) do t
    ftime[] = t
    #autolimits!(ax)
end