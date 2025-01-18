# Run this only on HPC
if Sys.CPU_NAME != "apple-m1"
    import Pkg
    Pkg.activate("../")
end
if Sys.CPU_NAME != "apple-m1"
    using CUDA
end

using Random
using Distributions
using Flux
using Zygote
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

device = Flux.gpu
try
    device = Flux.gpu_device()
    println("GPU device detected.")
catch
    println("No GPU device detected. Using CPU.")
end

##### EXPERIMENTAL CONDITIONS

TASKCAT = "reconstruction"
TASK = "recon80"

RANK::Int = 1
M, N, dataset_size = 80, 80, 1000
mat_size = M * N

knownentries = 1000
net_width::Int = 500

TURNS::Int = 5
EPOCHS = 1
MINIBATCH_SIZE = 64
n_loss = 10

##########
# Generate data and split between training, validation, and test sets
train_prop::Float64 = 0.8
val_prop::Float64 = 0.1
test_prop::Float64 = 0.1
ranks = [RANK]

function setup(M, N, r, seed; datatype=Float32)
    rng = Random.MersenneTwister(seed)
    #v = randn(rng, datatype, M, r) ./ sqrt(sqrt(Float32(r)))
    #A = v * v'
    A = (randn(rng, datatype, M, r) ./ sqrt(sqrt(Float32(r)))) * (randn(rng, datatype, r, N) ./ sqrt(sqrt(Float32(r))))
    return A * 0.1f0
end
Y = Array{Float32, 3}(undef, M, N, dataset_size)
for i in 1:dataset_size
    Y[:, :, i] = setup(M, N, RANK, 1131+i)
end

fixedmask = sensingmasks(M, N; k=knownentries, seed=9632)
mask_mat = masktuple2array(fixedmask)
@assert size(mask_mat) == (M, N)

X = matinput_setup(Y, net_width, M, N, dataset_size, knownentries, fixedmask)
Base.gc_live_bytes() / 1024^3

Xtrain, Xval, Xtest = train_val_test_split(X, train_prop, val_prop, test_prop)
X = nothing
Ytrain, Yval, Ytest = train_val_test_split(Y, train_prop, val_prop, test_prop)
Y = nothing
size(Xtrain), size(Ytrain)

println("Dataset created and split into training, validation, and test sets on cpu.")
##### INITIALIZE NETWORK

# Initialize the RNN model
unit_hidim = mat_size

activemodel = matnet(
    matrnn(mat_size, net_width, unit_hidim), 
    WMeanRecon(net_width)
)

println("Testing untrained model...")
activemodel() 
activemodel(Matrix(Float32.(Xtrain[1])))
reset!(activemodel)

activemodel = activemodel |> device
Flux.get_device(; verbose=true)
println("Model initialized and moved to gpu.")

##### DEFINE LOSS FUNCTIONS
myloss = recon_losses

##### TRAINING

# State optimizing rule
init_eta = 1e-4
decay = 0.7
s = Exp(start = init_eta, decay = decay)
println("Learning rate schedule:")
for (eta, epoch) in zip(s, 1:EPOCHS)
    println(" - Epoch $epoch: eta = $eta")
end
opt = Adam()
# Tree of states
opt_state = Flux.setup(opt, activemodel)
println("Optimizer and state initialized.")

traindataloader = Flux.DataLoader(
    (data=Xtrain, label=Ytrain), 
    batchsize=MINIBATCH_SIZE, 
    shuffle=true)
gpu_traindataloader = device(traindataloader)
println("Data loaded and moved to gpu.")

# STORE INITIAL METRICS
train_loss = Float32[]
val_loss = Float32[]
train_rmse = Float32[]
val_rmse = Float32[]
Whh_spectra = []
push!(Whh_spectra, eigvals(activemodel.rnn.cell.Whh |> cpu))

gt = false
initmetrics_train = myloss(activemodel, Xtrain[1:n_loss], Ytrain[:, :, 1:n_loss], mask_mat, gt; 
                           turns = TURNS, mode = "testing", incltrainloss = true)
initmetrics_val = myloss(activemodel, Xval[1:n_loss], Yval[:, :, 1:n_loss], mask_mat, gt; 
                         turns = TURNS, mode = "testing", incltrainloss = true)
push!(train_loss, initmetrics_train["l2"])
push!(train_rmse, initmetrics_train["RMSE"])
push!(val_loss, initmetrics_val["l2"])
push!(val_rmse, initmetrics_val["RMSE"])

starttime = time()
println("===================")
for (eta, epoch) in zip(s, 1:EPOCHS)
    reset!(activemodel)
    println("Commencing epoch $epoch (eta = $(round(eta, digits=6)))")
    Flux.adjust!(opt_state, eta = eta)
    # Initialize counters for gradient diagnostics
    mb, n, v, e = 1, 0, 0, 0
    # Iterate over minibatches
    for (x, y) in gpu_traindataloader
        # Pass twice over each minibatch (extra gradient learning)
        for _ in 1:2
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
    end
    push!(Whh_spectra, eigvals(activemodel.rnn.cell.Whh |> cpu))
    # Print a summary of the gradient diagnostics for the epoch
    diagnose_gradients(n, v, e)
    # Compute training metrics -- this is a very expensive operation because it involves a forward pass over the entire training set
    # so I take a subset of the training set to compute the metrics
    trainmetrics = myloss(activemodel, Xtrain[1:n_loss], Ytrain[:,:,1:n_loss], mask_mat, gt; 
                          turns = TURNS, mode = "testing", incltrainloss = true)
    push!(train_loss, trainmetrics["l2"])
    push!(train_rmse, trainmetrics["RMSE"])
    # Compute validation metrics
    valmetrics = myloss(activemodel, Xval[1:n_loss], Yval[:,:,1:n_loss], mask_mat, gt; 
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
rmsename = "Root mean squared reconstr. error / std (all entries)"
tasklab = "Reconstructing $(M)x$(M) rank-$(RANK) matrices from $(knownentries) of their entries"
taskfilename = "$(M)recon_rank$(RANK)"
modlabel = "Matrix Vanilla"

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

save("data/$(taskfilename)_$(modlabel)RNNwidth$(net_width)_$(TURNS)turns_knownentries.png", fig)

Whh = mactivemodel.rnn.cell.Whh |> cpu
bh = activemodel.rnn.cell.bh |> cpu
Wx_in = activemodel.rnn.cell.Wx_in |> cpu
bx_in = activemodel.rnn.cell.bx_in |> cpu
Wx_out = activemodel.rnn.cell.Wx_out |> cpu
bx_out = activemodel.rnn.cell.bx_out |> cpu
fig2 = Figure(size = (700, 800))
ax1 = Axis(fig2[1, 1], title = "Eigenvalues of Recurrent Weights", xlabel = "Real", ylabel = "Imaginary")
θ = LinRange(0, 2π, 1000)
scatter!(ax1, real(eigvals(Whh)), imag(eigvals(Whh)), color = :blue, alpha = 0.85, markersize = 5)
lines!(ax1, cos.(θ), sin.(θ), color = :black, linewidth = 2)
ax2 = Axis(fig2[1, 2], title = "Singular values of Recurrent Biases", xlabel = "Unit", ylabel = "Value")
scatter!(ax2, 1:net_width, svdvals(bh), color = :red, alpha = 0.85, markersize = 5)
ax3 = Axis(fig2[2, 1], title = "Singular values of Wx_in weights", xlabel = "Unit", ylabel = "Value")
scatter!(ax3, svdvals(Wx_in),
        color = :blue, alpha = 0.85, markersize = 5)
ax4 = Axis(fig2[2, 2], title = "Singular values of bx_in biases", xlabel = "Unit", ylabel = "Value")
scatter!(ax4, svdvals(bx_in), color = :purple, alpha = 0.85, markersize = 5)
ax5 = Axis(fig2[3, 1], title = "Singular values of Wx_out weights", xlabel = "Unit", ylabel = "Value")
scatter!(ax5, svdvals(Wx_out),
        color = :blue, alpha = 0.85, markersize = 5)
ax6 = Axis(fig2[3, 2], title = "Singular values of bx_out biases", xlabel = "Unit", ylabel = "Value")
scatter!(ax6, svdvals(bx_out), color = :purple, alpha = 0.85, markersize = 5)
fig2
save("data/$(taskfilename)_$(modlabel)RNNwidth$(net_width)_$(TURNS)turns_knownentries_learnedparamseigvals.png", fig2)

# Plot the same parameters, but as a heatmap of each matrix
fig3 = Figure(size = (600, 900))
cmap = :RdBu
ax1 = Axis(fig3[1, 1], title = "Recurrent Weights")
heatmap!(ax1, (Whh), colormap = cmap)
ax2 = Axis(fig3[1, 2], title = "Recurrent Biases")
heatmap!(ax2, (bh), colormap = cmap)
ax3 = Axis(fig3[2, 1], title = "Wx_in weights")
heatmap!(ax3, (Wx_in), colormap = cmap)
ax4 = Axis(fig3[2, 2], title = "bx_in biases")
heatmap!(ax4, (bx_in), colormap = cmap)
ax5 = Axis(fig3[3, 1], title = "Wx_out weights")
heatmap!(ax5, (Wx_out), colormap = cmap)
ax6 = Axis(fig3[3, 2], title = "bx_out biases")
heatmap!(ax6, (bx_out), colormap = cmap)
fig3
save("data/$(taskfilename)_$(modlabel)RNNwidth$(net_width)_$(TURNS)turns_knownentries_learnedparamshmaps.png", fig3)

include("inprogress/helpersWhh.jl")
g_end = adj_to_graph(Whh; threshold = 0.01)
figdegdist = plot_degree_distrib(g_end)
save("data/$(taskfilename)_$(modlabel)RNNwidth$(net_width)_$(TURNS)turns_knownentries_degreedist.png", figdegdist)

#p_i = hist(imag(eigvals(Whh)), bins = 70)
#p_r = hist(real(eigvals(Whh)), bins = 70)
#save("data/$(taskfilename)_$(modlabel)RNNwidth$(net_width)_$(TURNS)turns_knownentries_imageig.png", p_i)
#save("data/$(taskfilename)_$(modlabel)RNNwidth$(net_width)_$(TURNS)turns_knownentries_realeig.png", p_r)

# Create animation using each ploteigvals(jacobian_spectra[i]) as one frame
#using GLMakie
#GLMakie.activate!()

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
record(fw, "data/$(taskfilename)_$(modlabel)RNNwidth$(net_width)_$(TURNS)turns_knownentries_Whh.mp4", timestamps;
       framerate = framerate) do t
    ftime[] = t
    #autolimits!(ax)
end
