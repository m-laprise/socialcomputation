using Random
using Distributions
using Flux
using Zygote
using JLD2
using CairoMakie
using LinearAlgebra
#using IterTools

include("genrandommatrix.jl")
include("rnn_flux_cells.jl")

device = Flux.get_device(; verbose=true)

##########
TASK = "small_reconstruction"
TURNS = 30
VANILLA = false
INFERENCE_EXPERIMENT = false

if TASK == "small_reconstruction"
    samerank = false
end
##########

input_size = 0
net_width = 100

if TASK == "small_classification" || TASK == "random traces"
    output_size = 1
    m_binpred = Chain(
        rnn(input_size, net_width, h_init="randn"),
        Dense(net_width => output_size, sigmoid)
    )

    m_bfl = Chain(
        bfl(net_width, h_init="randn",
            basal_u = 0.001f0,
            damping = 0.75f0,
            gain = 0.5f0),
        Dense(net_width => output_size, sigmoid)
    )
end

if TASK == "small_reconstruction"
    output_size = 64
    m_binpred = Chain(
        rnn(input_size, net_width, h_init="randn"),
        Dense(net_width => output_size)
    )

    m_bfl = Chain(
        bfl(net_width, h_init="randn",
            basal_u = 0.001f0,
            damping = 0.75f0,
            gain = 0.5f0),
        Dense(net_width => output_size)
    )
end


# Flux.params(m_binpred)[1]
# Flux.params(m_binpred)[2]
# Flux.params(m_binpred)[3]
# Flux.params(m_binpred)[4]
# Flux.params(m_binpred)[5]

# Flux.params(m_bfl)[1]
# Flux.params(m_bfl)[2]
# Flux.params(m_bfl)[3]
# Flux.params(m_bfl)[4]
# Flux.params(m_bfl)[5]
# Flux.params(m_bfl)[6]
# Flux.params(m_bfl)[7]

# I   = randn(Float32, net_width) * 0.1f0
# y = m_binpred(I)[1]
# m_binpred.layers[1].state
# reset!(m_binpred.layers[1])

# y = m_bfl(I)[1]
# m_bfl.layers[1].state
# reset!(m_bfl.layers[1])

###########

dataset_size = 10000
train_size = 0.8
val_size = 0.1
test_size = 0.1

# Load training data
#X = load("data/rnn_flux_data.jld2","X")
Xmask = load("data/rnn_flux_data.jld2","Xmask")
Xtrace = load("data/rnn_flux_data.jld2","Xtrace")
Y = load("data/rnn_flux_data.jld2","Y")
mean(Y)
ranks = load("data/rnn_flux_data.jld2","ranks")

# Reshape row-observations to column-observations
#X = Float32.(permutedims(X, [2,3,1]))
Xmask = permutedims(Xmask, [2,3,1])
Xtrace = permutedims(Xtrace, [2,1])
Y = permutedims(Y, [2,1])


if TASK == "small_classification"
    X_data1 = zeros(Float32, (100, 10000))
    for i in 1:10000
        if Y[i] == 1 # Low rank
            X_data1[:,i] = pad_input(Float32.(vec(gen_matrix(8,8,1, seed=10+i))),100)
        else # Full rank
            X_data1[:,i] = pad_input(Float32.(vec(gen_matrix(8,8,8, seed=10+i))),100)
        end
    end
    X = X_data1
end

if TASK == "random traces"
    X = Xtrace
end

if TASK == "small_reconstruction"
    if samerank == true  # RANK 1 ONLY
        X_data2 = zeros(Float32, (64, 10000))
        for i in 1:10000
            X_data2[:,i] = Float32.(vec(gen_matrix(8,8,1, seed=10+i)))
        end
        ranks = [1 for i in Y]
        Y = copy(X_data2)
        # Randomly replace half of each column with zeros
        for i in 1:10000
            idxs = randperm(64)[1:32]
            X_data2[idxs,i] .= 0.0
        end
        X = X_data2
    else # RANK 1 or 2
        X_data2 = zeros(Float32, (64, 10000))
        for i in 1:10000
            if Y[i] == 1    # rank 1
                X_data2[:,i] = Float32.(vec(gen_matrix(8,8,1, seed=10+i)))
            else            # rank 2
                X_data2[:,i] = Float32.(vec(gen_matrix(8,8,2, seed=10+i)))
            end
        end
        ranks = [i == 1 ? 1 : 2 for i in Y]
        Y = copy(X_data2)
        # Randomly replace half of each column with zeros
        for i in 1:10000
            idxs = randperm(64)[1:32]
            X_data2[idxs,i] .= 0.0
        end
        X = X_data2
    end
end 

# train-val-test split
train_nb = Int(train_size * dataset_size)
val_nb = Int(val_size * dataset_size)
test_nb = Int(test_size * dataset_size)
train_idxs = 1:train_nb
val_idxs = train_nb+1:train_nb+val_nb
test_idxs = train_nb+val_nb+1:dataset_size

#Xmask_train, Xmask_val, Xmask_test = Xmask[:,:,train_idxs], Xmask[:,:,val_idxs], Xmask[:,:,test_idxs]
#Xtrace_train, Xtrace_val, Xtrace_test = Xtrace[:,train_idxs], Xtrace[:,val_idxs], Xtrace[:,test_idxs]
Xtrain, Xval, Xtest = X[:,train_idxs], X[:,val_idxs], X[:,test_idxs]
Ytrain, Yval, Ytest = Y[:,train_idxs], Y[:,val_idxs], Y[:,test_idxs]
ranks_train, ranks_val, ranks_test = ranks[train_idxs], ranks[val_idxs], ranks[test_idxs]

include("rnn_flux_lossfunctions.jl")

if TASK == "small_classification" || TASK == "random traces"
    myloss = logitbinarycrossent
    accuracy = classification_accuracy
elseif TASK == "small_reconstruction"
    myloss = recon_mse
    accuracy = spectral_distance
end

if TASK == "small_reconstruction"
    # For reference, compute the loss of a SOTA algorithm on this data.
    include("sota_matrix_completion.jl")
    function scaled_asd_performance(X_dataset, Y_dataset, maxiter)
        dataset_size = size(X_dataset, 2)
        mse_losses = zeros(Float32, dataset_size)
        spectral_dists = zeros(Float32, dataset_size)
        m, n = Int(sqrt(length(X_dataset[:, 1]))), Int(sqrt(length(X_dataset[:, 1])))
        opts = Dict(
                :rel_res_tol => 1e-5, 
                :maxit => maxiter,    
                :verbosity => false, 
                :rel_res_change_tol => 1e-4 
        )
        for i in 1:dataset_size
            X = reshape(X_dataset[:,i], m, n)
            Y = reshape(Y_dataset[:,i], m, n)
            r = rank(Y)
            I_idx, J_idx, knownentries = sparse2idx(Float64.(X))
            Y = Float64.(Y)
            soln, _ = ScaledASD(m, n, r, I_idx, J_idx, knownentries, opts; 
                                soln_only = true)
            mse_losses[i] = sum((Y .- soln) .^ 2)
            spectral_dists[i] = norm(svdvals(Y) .- svdvals(soln))
        end
        return mean(mse_losses), mean(spectral_dists)
    end
    #sota_train_mse, sota_train_spectraldist = scaled_asd_performance(Xtrain, Ytrain, 1500)
    #sota_val_mse, sota_val_spectraldist = scaled_asd_performance(Xval, Yval, 2000)
    sota_test_mse, sota_test_spectraldist = scaled_asd_performance(Xtest, Ytest, 3000)
end

#m_binpred((Xtrain[:,1]))
#m_binpred((Xtrain[:,1]))[1]
#m_binpred((Xtrain[:,2]))[1]
#Y[1], Y[2]
#a = myloss(m_binpred, (Xtrain[:,1]), Ytrain[:,1], turns = turns)
#b = myloss(m_binpred, (Xtrain[:,2]), Ytrain[:,2], turns = turns)
#c = myloss(m_binpred, (Xtrain[:,3]), Ytrain[:,3], turns = turns)
#ab = myloss(m_binpred, (Xtrain[:,2:3]), Ytrain[:,2:3], turns = turns)
#test1 = myloss(m_binpred, (Xtrain[:,1:20]), Ytrain[:,1:20], turns = turns)
#g = gradient(myloss, m_binpred, (Xtrain[:,1:20]), Ytrain[:,1:20])[1]


eta = 1e-3
#beta = (0.89, 0.995)
#decay = 0.1
#omega = 10
opt = OptimiserChain(
    #ClipNorm(omega), AdamW(eta, beta, decay)
    Adam(eta)
)

if VANILLA
    activemodel = m_binpred
    GATED = false
else
    activemodel = m_bfl
    GATED = true
end

opt_state = Flux.setup(opt, activemodel)
epochs = 50
minibatch_size = 64

traindataloader = Flux.DataLoader(
    (data=Xtrain, label=Ytrain), batchsize=minibatch_size, shuffle=true)

train_loss = Float32[]
val_loss = Float32[]
train_accuracy = Float32[]
val_accuracy = Float32[]

# Train using training data, plot training and validation accuracy and loss over training
# Using the Zygote package to compute gradients
reset!(activemodel.layers[1])
push!(train_loss, myloss(activemodel, Xtrain, Ytrain, turns = TURNS))
push!(train_accuracy, accuracy(activemodel, Xtrain, Ytrain, turns = TURNS))
push!(val_loss, myloss(activemodel, Xval, Yval, turns = TURNS))
push!(val_accuracy, accuracy(activemodel, Xval, Yval, turns = TURNS))

starttime = time()
for epoch in 1:epochs
    reset!(activemodel.layers[1])
    println("Commencing epoch $epoch")
    n, v, e = 0, 0, 0
    for (x, y) in traindataloader
        grads = gradient(myloss, activemodel, x, y)[1]
        _n, _v, _e = inspect_gradients(grads)
        n += _n
        v += _v
        e += _e
        Flux.update!(opt_state, activemodel, grads)
    end
    diagnose_gradients(n, v, e)
    #Flux.@withprogress Flux.train!(loss, activemodel, data, opt_state)
    # Compute training loss and accuracy
    push!(train_loss, myloss(activemodel, Xtrain, Ytrain, turns = TURNS))
    push!(train_accuracy, accuracy(activemodel, Xtrain, Ytrain, turns = TURNS))
    # Compute validation loss and accuracy
    push!(val_loss, myloss(activemodel, Xval, Yval, turns = TURNS))
    push!(val_accuracy, accuracy(activemodel, Xval, Yval, turns = TURNS))
    println("Epoch $epoch: Train loss: $(train_loss[end]), Train accuracy: $(train_accuracy[end])")
end
endtime = time()
# training time in minutes
println("Training time: $(round((endtime - starttime) / 60, digits=2)) minutes")
# Compute test accuracy
test_accuracy = accuracy(activemodel, Xtest, Ytest, turns = TURNS)
println("Test accuracy: $test_accuracy")
test_loss = myloss(activemodel, Xtest, Ytest, turns = TURNS)
println("Test loss: $test_loss")


if TASK == "small_reconstruction"
    lossname = "Mean squared reconstruction error"
    accuracyname = "Mean norm of spectral distance (singular values)"
else
    lossname = "Binary cross-entropy loss"
    accuracyname = "Classification accuracy"
end
if TASK == "small_reconstruction"
    tasklab = "Reconstructing 8x8 rank 1 or 2 matrices from half of their entries"
    taskfilename = "8by8recon"
elseif TASK == "small_classification"
    tasklab = "Classifying 8x8 matrices as full rank or rank 1"
    taskfilename = "8by8class"
elseif TASK == "random traces"
    tasklab = "Classifying 250x250 matrices as low/high rank from 50 traces of random projections"
    taskfilename = "250by250tracesclass"
end

fig = Figure(size = (820, 450))
epochs = length(train_loss) - 1
ax_loss = Axis(fig[1, 1], xlabel = "Epochs", ylabel = "Loss", title = lossname)
lines!(ax_loss, 1:epochs+1, train_loss, color = :blue, label = "Training")
lines!(ax_loss, 1:epochs+1, val_loss, color = :red, label = "Validation")
hlines!(ax_loss, [test_loss], color = :green, linestyle = :dash, label = "Final Test")
#ylims!(ax_loss, 35.0, 170.0)
axislegend(ax_loss, backgroundcolor = :transparent)
ax_acc = Axis(fig[1, 2], xlabel = "Epochs", ylabel = "", title = accuracyname)
lines!(ax_acc, 1:epochs+1, train_accuracy, color = :blue, label = "Training")
lines!(ax_acc, 1:epochs+1, val_accuracy, color = :red, label = "Validation")
hlines!(ax_acc, [test_accuracy], color = :green, linestyle = :dash, label = "Final Test")
#ylims!(ax_acc, 4.0, 7.0)
axislegend(ax_acc, backgroundcolor = :transparent, position = :rt)
# Add a title to the top of the figure
modlabel = GATED ? "BFL" : "Vanilla"
Label(
    fig[begin-1, 1:2],
    "$(tasklab)\n$(modlabel) RNN of $net_width units, $TURNS dynamic steps",
    fontsize = 20,
    padding = (0, 0, 0, 0),
)
# Add notes to the bottom of the figure
Label(
    fig[end+1, 1:2],
    "Optimizer: Adam(eta=$eta)\n"*
    "Training time: $(round((endtime - starttime) / 60, digits=2)) minutes; "*
    "Test loss: $(round(test_loss,digits=2)). Test spectral distance: $(round(test_accuracy,digits=2)).\n"*
    "Scaled ASD with known rank on test set: loss of $(round(sota_test_mse,digits=2)); spectral distance of $(round(sota_test_spectraldist,digits=2)).",
    #"Optimizer: AdamW(eta=$eta, beta=$beta, decay=$decay)\nTraining time: $(round((endtime - starttime) / 60, digits=2)) minutes; Test accuracy: $(round(test_accuracy,digits=2)).",
    fontsize = 14,
    padding = (0, 0, 0, 0),
)
fig

save("data/$(modlabel)RNNwidth100_$(taskfilename)_$(TURNS)turns_rank1only.png", fig)



# Save a copy of the final recurrent weights Whh
Whh = Flux.params(activemodel.layers[1].cell)[1]

bs = Flux.params(activemodel.layers[1].cell)[2]
if GATED
    damprates = Flux.params(activemodel.layers[1].cell)[3]
    gains = Flux.params(activemodel.layers[1].cell)[4]
end
eigvals_Whh = eigvals(Whh)

if GATED
    height = 500
else
    height = 400
end
fig2 = Figure(size = (700, height))
ax1 = Axis(fig2[1, 1], title = "Eigenvalues of Recurrent Weights", xlabel = "Real", ylabel = "Imaginary")
# Plot the unit circle
θ = LinRange(0, 2π, 1000)
circle_x = cos.(θ)
circle_y = sin.(θ)
lines!(ax1, circle_x, circle_y, color = :black)
# Plot the eigenvalues
scatter!(ax1, real(eigvals_Whh), imag(eigvals_Whh), color = :blue)
# Plot on different axes lollipop graphs of bs, damprates, gains
ax2 = Axis(fig2[1, 2], title = "Biases", xlabel = "Unit", ylabel = "Value")
barplot!(ax2, 1:net_width, bs, color = :red)
if GATED
    ax3 = Axis(fig2[2, 1], title = "Damping Rates", xlabel = "Unit", ylabel = "Value")
    barplot!(ax3, 1:net_width, damprates, color = :green)
    ax4 = Axis(fig2[2, 2], title = "Gains", xlabel = "Unit", ylabel = "Value")
    barplot!(ax4, 1:net_width, gains, color = :purple)
end
fig2

save("data/$(modlabel)RNNwidth100_$(taskfilename)_$(TURNS)turns_Whh.jld2", "Whh", Whh)
save("data/$(modlabel)RNNwidth100_$(taskfilename)_$(TURNS)turns_learnedparams.png", fig2)



#if INFERENCE_EXPERIMENT

    k = 151
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
    axislegend(ax2, backgroundcolor = :transparent)
    Label(fig3[begin-1, 1:2], 
        "$(tasklab)\n$(modlabel) RNN of $net_width units, trained with $TURNS forward passes\nRecurrent inference on test set", 
        fontsize = 16)
    Label(fig3[end+1, 1:2], 
        "EOT stands for end of training. Spectral distance comparison is not shown because\n"*
        "the Scaled ASD algorithm uses the true ranks of the underlying ground truth, while\n"*
        "the RNN has no rank information.", fontsize = 12)
    fig3

    #save("data/$(modlabel)RNNwidth100_$(taskfilename)_$(TURNS)turns_inferencetests.png", fig3)
#end