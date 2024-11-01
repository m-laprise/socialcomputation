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

#device = Flux.get_device(; verbose=true)

##########

taskcats = ["classification", "reconstruction"]
measurecats = ["masks", "traces", "blocks"]
tasks = ["classif1a", "classif1b", "classif2a", "classif2b", 
         "recon32", "recon64", "recon128", "recon256",
         "sparse200a", "sparse200b"]
datasetnames = ["lr_c1", "lr_c2", 
                "lr_r_32", "lr_r_64", "lr_r_128", "lr_r_256", 
                "sparse_200"]

DATASETNAME = datasetnames[3]
TASKCAT = taskcats[2]
MEASCAT = measurecats[1]
TASK = tasks[5]
TURNS = 10
VANILLA = true

INFERENCE_EXPERIMENT = false
SOCGRAPHINIT = false

net_width = 100

##########

train_prop = 0.8
val_prop = 0.1
test_prop = 0.1

data = load("data/rnn_firstexpdata.jld2", DATASETNAME)
m, n, dataset_size = size(data["X"])

# Create label data
if TASKCAT == "reconstruction"
    Y = data["X"]
elseif TASKCAT == "classification"
    if TASK == "classif1a" || TASK == "classif1b"
        Y = data["ranks"] .> 100
    elseif TASK == "classif2a" || TASK == "classif2b"
        ranks = data["ranks"]
        # Convert vector of possible ranks to multi-class one-hot encoded matrix
        nb_classes = length(unique(ranks))
        ordered_classes = sort(unique(ranks))
        Y = zeros(Int, nb_classes, dataset_size)
        for i in 1:dataset_size
            Y[findall(x -> x == ranks[i], ordered_classes)[1], i] = 1
        end
    end
end

# Create input data
if MEASCAT == "masks"
    masks = sensingmasks(m, n; k=net_width, seed=9632)
    X = zeros(Float32, net_width, dataset_size)
    for k in 1:dataset_size
        for l in 1:net_width
            i, j = masks[l]
            X[l, k] = Y[i, j, k]
        end
    end
    #= X = zeros(Float32, net_width, 3, dataset_size)
    for k in 1:dataset_size
        for l in 1:net_width
            i, j = masks[l]
            X[l, 1, k] = Y[i, j, k]
            X[l, 2, k] = i
            X[l, 3, k] = j
        end
    end =#
elseif MEASCAT == "traces" || MEASCAT == "blocks"
    error("Not implemented yet.")
end

Xtrain, Xval, Xtest = train_val_test_split(X, train_prop, val_prop, test_prop)
Ytrain, Yval, Ytest = train_val_test_split(Y, train_prop, val_prop, test_prop)
if TASKCAT == "reconstruction"
    ranks = unique(data["ranks"])
    size(ranks)
else
    ranks_train, ranks_val, ranks_test = train_val_test_split(data["ranks"], train_prop, val_prop, test_prop)
    size(ranks_train)
end
size(masks), size(Xtrain), size(Ytrain)
##########

input_size = 0

Whh_init = nothing
#Whh_init = load("data/Whh_init.jld2", "Whh_init")
if SOCGRAPHINIT
    include("inprogress/helpersWhh.jl")
    #g = init_socgraph("Barabasi-Albert", net_width, 3, 9632)
    #g = init_socgraph("Erdos-Renyi", net_width, 3, 9632)
    g = init_socgraph("Watts-Strogatz", net_width, 3, 9632)
    adj = Float32.(graph_to_adj(g))
    Whh_init = adj .+ 0.5*(randn(Float32, net_width, net_width) / sqrt(Float32(net_width)))
    #print_socgraph_descr(g)
    #plot_socgraph(g)
    #plot_degree_distrib(g)
end

if TASKCAT == "classification"
    
    if TASK == "classif1a" || TASK == "classif1b"
        output_size = 1
    elseif TASK == "classif2a" || TASK == "classif2b"
        output_size = nb_classes
    end

    m_binpred = Chain(
        rnn(input_size, net_width, 
            Whh_init = Whh_init, 
            h_init="randn"),
        Dense(net_width => output_size, sigmoid)
    )

    m_bfl = Chain(
        bfl(net_width, 
            Whh_init = Whh_init,
            h_init="randn", 
            basal_u = 0.001f0,
            gain = 0.5f0),
        Dense(net_width => output_size, sigmoid)
    )

elseif TASKCAT == "reconstruction"
    
    output_size = m * n

    m_binpred = Chain(
        rnn(input_size, net_width, 
            Whh_init = Whh_init, 
            h_init="randn"),
        Dense(net_width => output_size)
    )
    m_bfl = Chain(
        bfl(net_width, 
            Whh_init = Whh_init, 
            h_init="randn",
            basal_u = 0.001f0,
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
# y = m_bfl(I)[1]
# m_bfl.layers[1].state
# reset!(m_bfl.layers[1])

###########

if TASKCAT == "classification" 
    myloss = logitbinarycrossent
    accuracy = classification_accuracy
elseif TASKCAT == "reconstruction"
    myloss = recon_mse
    accuracy = spectral_distance
end

if TASKCAT == "reconstruction"
    # For reference, compute the loss of a non-distributed algorithm on this data.
    include("sota_matrix_completion.jl")
    opts = Dict(
        :rel_res_tol => 1e-5, 
        :maxit => 1000,    
        :verbosity => false, 
        :rel_res_change_tol => 1e-4
    )
    I_idx = zeros(Int, net_width)
    J_idx = zeros(Int, net_width)
    for i in 1:net_width
        I_idx[i], J_idx[i] = masks[i]
    end
    #sota_train_mse, sota_train_spectraldist = scaled_asd_performance(Xtrain, Ytrain, I_idx, J_idx, opts)
    #sota_val_mse, sota_val_spectraldist = scaled_asd_performance(Xval, Yval, I_idx, J_idx, opts)
    sota_test_mse, sota_test_spectraldist = scaled_asd_performance(Xtest, Ytest, I_idx, J_idx, opts, 8)
end

#m_binpred((Xtrain[:,1]))
#m_binpred((Xtrain[:,1]))[1]
#m_binpred((Xtrain[:,2]))[1]
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

jacobian_spectra = []
# Train using training data, plot training and validation accuracy and loss over training
# Using the Zygote package to compute gradients
reset!(activemodel.layers[1])

push!(train_loss, myloss(activemodel, Xtrain, Ytrain, turns = TURNS))
push!(train_accuracy, accuracy(activemodel, Xtrain, Ytrain, turns = TURNS) )
push!(val_loss, myloss(activemodel, Xval, Yval, turns = TURNS))
push!(val_accuracy, accuracy(activemodel, Xval, Yval, turns = TURNS) )

starttime = time()
for epoch in 1:epochs
    reset!(activemodel.layers[1])
    println("Commencing epoch $epoch")
    n, v, e = 0, 0, 0
    for (x, y) in traindataloader
        grads = gradient(myloss, activemodel, x, y)[1]
        _n, _v, _e = inspect_gradients(grads)
        J = getjacobian(activemodel; wrt = "state")
        push!(jacobian_spectra, eigvals(J))
        n += _n
        v += _v
        e += _e
        Flux.update!(opt_state, activemodel, grads)
    end
    diagnose_gradients(n, v, e)
    #Flux.@withprogress Flux.train!(loss, activemodel, data, opt_state)
    # Compute training loss and accuracy
    push!(train_loss, myloss(activemodel, Xtrain, Ytrain, turns = TURNS))
    push!(train_accuracy, accuracy(activemodel, Xtrain, Ytrain, turns = TURNS) )
    # Compute validation loss and accuracy
    push!(val_loss, myloss(activemodel, Xval, Yval, turns = TURNS))
    push!(val_accuracy, accuracy(activemodel, Xval, Yval, turns = TURNS) )
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


if TASKCAT == "reconstruction"
    lossname = "Mean squared reconstruction error"
    accuracyname = "Mean norm of spectral distance (singular values)"
else
    lossname = "Binary cross-entropy loss"
    accuracyname = "Classification accuracy"
end
if TASK == "recon32"
    tasklab = "Reconstructing 32x32 rank 8 matrices from 100 of their entries"
    taskfilename = "32recon"
#= elseif TASK == "small_classification"
    tasklab = "Classifying 8x8 matrices as full rank or rank 1"
    taskfilename = "8by8class"
elseif TASK == "random traces"
    tasklab = "Classifying 250x250 matrices as low/high rank from 50 traces of random projections"
    taskfilename = "250by250tracesclass" =#
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

save("data/$(modlabel)RNNwidth100_$(taskfilename)_$(TURNS)turns.png", fig)

Whh = Flux.params(activemodel.layers[1].cell)[1]
bs = Flux.params(activemodel.layers[1].cell)[2]

if GATED
    damprates = Flux.params(activemodel.layers[1].cell)[3]
    gains = Flux.params(activemodel.layers[1].cell)[4]
end
if GATED
    height = 500
else
    height = 400
end

fig2 = Figure(size = (700, height))
ax1 = Axis(fig2[1, 1], title = "Eigenvalues of Recurrent Weights", xlabel = "Real", ylabel = "Imaginary")
ploteigvals!(ax1, Whh)
ax2 = Axis(fig2[1, 2], title = "Biases", xlabel = "Unit", ylabel = "Value")
barplot!(ax2, 1:net_width, bs, color = :red)
if GATED
    #ax3 = Axis(fig2[2, 1], title = "Damping Rates", xlabel = "Unit", ylabel = "Value")
    #barplot!(ax3, 1:net_width, damprates, color = :green)
    ax4 = Axis(fig2[2, 2], title = "Gains", xlabel = "Unit", ylabel = "Value")
    barplot!(ax4, 1:net_width, gains, color = :purple)
end
fig2

save("data/$(modlabel)RNNwidth100_$(taskfilename)_$(TURNS)turns_Whh.jld2", "Whh", Whh)
save("data/$(modlabel)RNNwidth100_$(taskfilename)_$(TURNS)turns_learnedparams.png", fig2)

include("inprogress/helpersWhh.jl")
g_end = adj_to_graph(Whh)
print_socgraph_descr(g_end)
plot_degree_distrib(g_end)


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
    axislegend(ax2, backgroundcolor = :transparent)
    Label(fig3[begin-1, 1:2], 
        "$(tasklab)\n$(modlabel) RNN of $net_width units, trained with $TURNS forward passes\nRecurrent inference on test set", 
        fontsize = 16)
    Label(fig3[end+1, 1:2], 
        "EOT stands for end of training. Spectral distance comparison is not shown because\n"*
        "the Scaled ASD algorithm uses the true ranks of the underlying ground truth, while\n"*
        "the RNN has no rank information.", fontsize = 12)
    fig3

    save("data/$(modlabel)RNNwidth100_$(taskfilename)_$(TURNS)turns_inferencetests_WSinit.png", fig3)
end
