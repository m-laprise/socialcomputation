# Run this only on HPC
if Sys.CPU_NAME != "apple-m1"
    import Pkg
    Pkg.activate("../")
end

using CUDA, Adapt
using Random
using Distributions
using Flux
using Zygote, Enzyme
#set_runtime_activity(Enzyme.Reverse, true)
#Enzyme.Compiler.VERBOSE_ERRORS[] = true
using CairoMakie
using LinearAlgebra
using ParameterSchedulers
#using ParameterSchedulers: Scheduler

include("genrandommatrix.jl")
include("rnn_cells.jl")
include("customlossfunctions.jl")
include("plot_utils.jl")
include("train_utils.jl")
include("train_setup.jl")

device = Flux.gpu
try
    device = Flux.gpu_device()
catch
    @info("No GPU device detected. Using CPU.")
end

##### EXPERIMENTAL CONDITIONS

TASKCAT = "reconstruction"
TASK = "recon80"

const RANK::Int = 1
const M::Int, N::Int, dataset_size::Int = 64, 64, 1000
const mat_size::Int = M * N

knownentries::Int = 1500
net_width::Int = 400

TURNS::Int = 5
EPOCHS = 1
const MINIBATCH_SIZE::Int = 32
n_loss::Int = 10

##########
# Generate data and split between training, validation, and test sets
const train_prop::Float64 = 0.8
const val_prop::Float64 = 0.1
const test_prop::Float64 = 0.1
const ranks = [RANK]

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
mask_mat = Float32.(masktuple2array(fixedmask))
@assert size(mask_mat) == (M, N)
X = matinput_setup(
    Y, net_width, M, N, dataset_size, knownentries, fixedmask)
@info("Memory usage after data generation: ", Base.gc_live_bytes() / 1024^3)

Xtrain, Xval, Xtest = train_val_test_split(X, train_prop, val_prop, test_prop)
X = nothing
Ytrain, Yval, Ytest = train_val_test_split(Y, train_prop, val_prop, test_prop)
Y = nothing
size(Xtrain), size(Ytrain)

@info("Dataset created and split into training, validation, and test sets on cpu.")
##### INITIALIZE NETWORK

# Initialize the RNN model
unit_hidim = mat_size

cpu_activemodel = matnet(
    matrnn_cell(mat_size, net_width, unit_hidim), 
    WMeanRecon(net_width)
)

@info("Testing untrained model on cpu...")
cpu_activemodel()
cpu_activemodel(Xtrain[1])
cpu_activemodel(; selfreset = true)
cpu_activemodel(Xtrain[1]; selfreset = true)
reset!(cpu_activemodel)

#recon_losses(cpu_activemodel, Xtrain[1:2], Ytrain[:, :, 1:2], mask_mat; 
#             turns = TURNS, mode = "testing")

activemodel = cpu_activemodel |> device
@info("Model initialized and moved to device.")

if CUDA.functional()
    @info("Testing untrained model on gpu...")
    sum(activemodel())
    sum(activemodel(device(Xtrain[1])))
    sum(activemodel(device(Xtrain[1]); selfreset = true))
    reset!(activemodel)
end

##### DEFINE LOSS FUNCTIONS
myloss = spectrum_penalized_l2

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
@info("Optimizer and state initialized.")

traindataloader = Flux.DataLoader(
    (data=Xtrain, label=Ytrain), 
    batchsize=MINIBATCH_SIZE, 
    shuffle=true)

if CUDA.functional()
    # For X, use stack to convert Vector{SparseMatrixCSC} (which is not bitstypes) 
    # to a 3D tensor of type Array{Float32, 3} that can be stored contiguously on GPU
    gpu_Xtrain, gpu_Ytrain = device(stack(Xtrain)), device(Ytrain)
    isbitstype(eltype(gpu_Xtrain))
    gpu_Xval, gpu_Yval = device(stack(Xval)), device(Yval)
    gpu_Xtest, gpu_Ytest = device(stack(Xtest)), device(Ytest)
    gpu_traindataloader = Flux.DataLoader(
        (data=gpu_Xtrain, label=gpu_Ytrain),
        batchsize=MINIBATCH_SIZE,
        shuffle=true)
    gpu_mask_mat = device(mask_mat)
end
@info("Data loaded and moved to device.")

# STORE INITIAL METRICS
train_loss = Float32[]
val_loss = Float32[]
train_rmse = Float32[]
val_rmse = Float32[]
Whh_spectra = []
push!(Whh_spectra, eigvals(activemodel.rnn.cell.Whh |> cpu))

#gt = false
# Compute the initial training and validation loss with forward passes on GPU and store it back to CPU
initloss_train = myloss(activemodel, gpu_Xtrain[:, :, 1:n_loss], gpu_Ytrain[:, :, 1:n_loss], gpu_mask_mat; 
                        turns = TURNS)
initRMSE_train = allentriesRMSE(activemodel, gpu_Xtrain[:, :, 1:n_loss], gpu_Ytrain[:, :, 1:n_loss], gpu_mask_mat; 
                                turns = TURNS)
initloss_val = myloss(activemodel, gpu_Xval[:, :, 1:n_loss], gpu_Yval[:, :, 1:n_loss], gpu_mask_mat; 
                         turns = TURNS)
initRMSE_val = allentriesRMSE(activemodel, gpu_Xval[:, :, 1:n_loss], gpu_Yval[:, :, 1:n_loss], gpu_mask_mat; 
                              turns = TURNS)
push!(train_loss, initloss_train)
push!(train_rmse, initRMSE_train)
push!(val_loss, initloss_val)
push!(val_rmse, initRMSE_val)

#smallmodel = matrnn(mat_size, net_width, unit_hidim) |> device

a = [(x,y) for (x,y) in gpu_traindataloader]
x, y = a[1]
x = x[:,:,1:2]
y = y[:,:,1:2]
reset!(activemodel)

ref_loss, ref_grads = Flux.withgradient(myloss, activemodel, x, y, device(mask_mat))

InteractiveUtils.@code_warntype activemodel(x)
# Some union types could not be removed
Enzyme.API.strictAliasing!(false)
# Precompile gradient calculation
#=To test on CPU that Enzyme will work, 
ensure that InteractiveUtils.@code_warntype reports no type instability =#

@info("Starting to precompile reverse mode autodiff...")
starttime = time()
grads = autodiff(set_runtime_activity(Enzyme.Reverse), 
    myloss, Duplicated(activemodel), 
    Const(x), Const(y), Const(gpu_mask_mat))
endtime = time()
@info("Precompiled in $(round((endtime - starttime) / 60, digits=2)) minutes")


#*NOTE*: Must modify training loop to reflect training loss and other loss coming from different functions.
starttime = time()
println("===================")
for (eta, epoch) in zip(s, 1:EPOCHS)
    reset!(activemodel)
    println("Commencing epoch $epoch (eta = $(round(eta, digits=6)))")
    @info("Memory usage: ", Base.gc_live_bytes() / 1024^3)
    Flux.adjust!(opt_state, eta = eta)
    # Initialize counters for gradient diagnostics
    mb, n, v, e = 1, 0, 0, 0
    # Iterate over minibatches
    for (x, y) in gpu_traindataloader
        # Pass twice over each minibatch (extra gradient learning)
        for _ in 1:2
            # Forward pass (to compute the loss) and backward pass (to compute the gradients)
            train_loss_value, grads = Flux.withgradient(myloss, activemodel, x, y, mask_mat)
            GC.gc()
            @info("Memory usage: ", Base.gc_live_bytes() / 1024^3)
            # During training, use the backward pass to store the training loss after the previous epoch
            push!(train_loss, train_loss_value)
            # Diagnose the gradients
            _n, _v, _e = inspect_gradients(grads[1])
            n += _n
            v += _v
            e += _e
            # Detect loss of Inf or NaN. Print a warning, and then skip update!
            if !isfinite(train_loss_value)
                @warn "Loss is $val on minibatch $(epoch)--$(mb)" 
                mb += 1
                continue
            end
            # Use the optimizer and grads to update the trainable parameters; update the optimizer states
            Flux.update!(opt_state, activemodel, grads[1])
            GC.gc()
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
    trainloss = myloss(activemodel, gpu_Xtrain[1:n_loss], gpu_Ytrain[:,:,1:n_loss], mask_mat; 
                          turns = TURNS, mode = "testing")
    trainRMSE = allentriesRMSE(activemodel, gpu_Xtrain[1:n_loss], gpu_Ytrain[:,:,1:n_loss], mask_mat; 
                               turns = TURNS)
    push!(train_loss, trainloss)
    push!(train_rmse, trainRMSE)
    # Compute validation metrics
    valloss = myloss(activemodel, gpu_Xval[1:n_loss], gpu_Yval[:,:,1:n_loss], mask_mat; 
                        turns = TURNS, mode = "testing")
    valRMSE = allentriesRMSE(activemodel, gpu_Xval[1:n_loss], gpu_Yval[:,:,1:n_loss], mask_mat; 
                             turns = TURNS)
    push!(val_loss, valloss)
    push!(val_rmse, valRMSE)
    println("Epoch $epoch: Train loss: $(train_loss[end]); train RMSE: $(train_rmse[end])")
    println("Epoch $epoch: Val loss: $(val_loss[end]); val RMSE: $(val_rmse[end])")
    # Check if validation loss has increased for 2 epochs in a row; if so, stop training
    if length(val_loss) > 2
        if val_loss[end] > val_loss[end-1] && val_loss[end-1] > val_loss[end-2]
            @warn("Early stopping at epoch $epoch")
            break
        end
    end
end
endtime = time()
# training time in minutes
println("Training time: $(round((endtime - starttime) / 60, digits=2)) minutes")
# Assess on testing set
testloss = myloss(activemodel, gpu_Xtest, gpu_Ytest, mask_mat; 
                     turns = TURNS, mode = "testing")
testRMSE = allentriesRMSE(activemodel, gpu_Xtest, gpu_Ytest, mask_mat; 
                          turns = TURNS)
println("Test RMSE: $(testRMSE)")
println("Test loss: $(testloss)")

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
