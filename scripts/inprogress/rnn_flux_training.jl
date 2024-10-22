using Random
using Distributions
using Flux
using Zygote
using JLD2
using CairoMakie
using LinearAlgebra
#using IterTools

include("rnn_flux_cells.jl")

device = Flux.get_device(; verbose=true)

##########
input_size = 0
net_width = 100
output_size = 1

m_binpred = Chain(
    rnn(input_size, net_width, h_init="randn"),
    Dense(net_width => output_size, sigmoid)
)

#Flux.params(m_binpred)[1]
#Flux.params(m_binpred)[2]
#Flux.params(m_binpred)[3]
#Flux.params(m_binpred)[4]
#Flux.params(m_binpred)[5]

#I   = randn(Float32, net_width) * 0.1f0
#y = m_binpred(I)
#m_binpred.layers[1].state
#reset!(m_binpred.layers[1])

###########


dataset_size = 10000
train_size = 0.8
val_size = 0.1
test_size = 0.1

# Load training data
X = load("data/rnn_flux_data.jld2","X")
Xmask = load("data/rnn_flux_data.jld2","Xmask")
Xtrace = load("data/rnn_flux_data.jld2","Xtrace")
Y = load("data/rnn_flux_data.jld2","Y")
mean(Y)
ranks = load("data/rnn_flux_data.jld2","ranks")

# train-val-test split
train_nb = Int(train_size * dataset_size)
val_nb = Int(val_size * dataset_size)
test_nb = Int(test_size * dataset_size)
train_idxs = 1:train_nb
val_idxs = train_nb+1:train_nb+val_nb
test_idxs = train_nb+val_nb+1:dataset_size

Xmask_train, Xmask_val, Xmask_test = Xmask[train_idxs,:,:], Xmask[val_idxs,:,:], Xmask[test_idxs,:,:]
Xtrace_train, Xtrace_val, Xtrace_test = Xtrace[train_idxs,:], Xtrace[val_idxs,:], Xtrace[test_idxs,:]
Ytrain, Yval, Ytest = Y[train_idxs,:], Y[val_idxs,:], Y[test_idxs,:]
ranks_train, ranks_val, ranks_test = ranks[train_idxs], ranks[val_idxs], ranks[test_idxs]

"""
    loss(m, x, y)
    Many-to-one loss function for binary classification. Each input is a vector, each output is a binary label.
    Hence x should be a sliceobject to iterate over input vectors. y is the vector of labels.
"""
function loss(m, xs, ys; turns = 10)
    reset!(m.layers[1])
    net_width = length(state(m.layers[1]))
    for _ in 1:turns
        m(pad_input(eachrow(xs), net_width))
    end
    ys_hat = [m(pad_input(x, net_width))[1] for x in xs]
    #println(y_hat)
    losses = @.((1 - ys) * ys_hat - logσ(ys_hat))
    #println("Size losses: $(size(losses))")
    return mean(losses)
end

turns = 10

m_binpred(pad_input(Xtrace_train[1,:], net_width))
m_binpred(pad_input(Xtrace_train[1,:], net_width))[1]
m_binpred(pad_input(Xtrace_train[2,:], net_width))[1]
Y[1], Y[2]
a = loss(m_binpred, eachcol(Xtrace_train[1,:]), Y[1,:], turns = turns)
b = loss(m_binpred, eachcol(Xtrace_train[2,:]), Y[2,:], turns = turns)
ab = loss(m_binpred, eachrow(Xtrace_train[1:2,:]), Y[1:2,:], turns = turns)
test1 = loss(m_binpred, eachrow(Xtrace_train[1:20,:]), Y[1:20,:], turns = turns)
g = gradient(loss, m_binpred, eachrow(Xtrace_train[1:20,:]), Y[1:20,:])[1]


#using BenchmarkTools
#@benchmark test1 = loss(m_binpred, eachrow(Xtrace_train[1:20,:]), Y[1:20,:])
#@benchmark test2 = loss2(m_binpred, eachrow(Xtrace_train[1:20,:]), Y[1:20,:])
#@benchmark g = gradient(loss, m_binpred, eachrow(Xtrace_train[1:20,:]), Y[1:20,:])[1]
#@benchmark g2 = gradient(loss2, m_binpred, eachrow(Xtrace_train[1:20,:]), Y[1:20,:])[1]

function accuracy(m, xs, ys; turns = 10)
    reset!(m.layers[1])
    net_width = length(state(m.layers[1]))
    for _ in 1:turns
        m(pad_input(xs, net_width))
    end
    probs = [m(pad_input(x, net_width))[1] for x in xs]
    labels = Bool.(ys)
    preds = probs .> 0.5f0
    whether_correct = preds .== labels
    return sum(whether_correct) / length(labels)
end

opt = OptimiserChain(ClipGrad(10), Adam(1e-3))
opt_state = Flux.setup(opt, m_binpred)
epochs = 50
minibatch_size = 64

train_loss = Float32[]
val_loss = Float32[]
train_accuracy = Float32[]
val_accuracy = Float32[]

# Train using training data, plot training and validation accuracy and loss over training
# Using the Zygote package to compute gradients
traindataloader = Flux.DataLoader(
    (data=Xtrace_train', label=Ytrain'), batchsize=minibatch_size, shuffle=true)

reset!(m_binpred.layers[1])
push!(train_loss, loss(m_binpred, eachrow(Xtrace_train), Ytrain, turns = turns))
push!(train_accuracy, accuracy(m_binpred, eachrow(Xtrace_train), Ytrain, turns = turns))
push!(val_loss, loss(m_binpred, eachrow(Xtrace_val), Yval, turns = turns))
push!(val_accuracy, accuracy(m_binpred, eachrow(Xtrace_val), Yval, turns = turns))

starttime = time()
for epoch in 1:10
    reset!(m_binpred.layers[1])
    println("Commencing epoch $epoch")
    for (x, y) in traindataloader
        grads = gradient(loss, m_binpred, x, y)[1]
        Flux.update!(opt_state, m_binpred, grads)
    end
    #Flux.@withprogress Flux.train!(loss, m_binpred, data, opt_state)
    # Compute training loss and accuracy
    push!(train_loss, loss(m_binpred, eachrow(Xtrace_train), Ytrain, turns = turns))
    push!(train_accuracy, accuracy(m_binpred, eachrow(Xtrace_train), Ytrain, turns = turns))
    # Compute validation loss and accuracy
    push!(val_loss, loss(m_binpred, eachrow(Xtrace_val), Yval, turns = turns))
    push!(val_accuracy, accuracy(m_binpred, eachrow(Xtrace_val), Yval, turns = turns))
    println("Epoch $epoch: Train loss: $(train_loss[end]), Train accuracy: $(train_accuracy[end])")
end
endtime = time()
# training time in minutes
println("Training time: $(round((endtime - starttime) / 60, digits=2)) minutes")
# Compute test accuracy
test_accuracy = accuracy(m_binpred, eachrow(Xtrace_test), Ytest, turns = turns)
println("Test accuracy: $test_accuracy")
test_loss = loss(m_binpred, eachrow(Xtrace_test), Ytest, turns = turns)


fig = Figure(size = (800, 400))
epochs = length(train_loss) - 1
ax_loss = Axis(fig[1, 1], xlabel = "Epochs", ylabel = "Loss", title = "Binary Cross-Entropy Loss")
lines!(ax_loss, 1:epochs+1, train_loss, color = :blue, label = "Training Loss")
lines!(ax_loss, 1:epochs+1, val_loss, color = :red, label = "Validation Loss")
hlines!(ax_loss, [test_loss], color = :green, linestyle = :dash, label = "Final Test Loss")
axislegend(ax_loss, backgroundcolor = :transparent)
ax_acc = Axis(fig[1, 2], xlabel = "Epochs", ylabel = "Accuracy", title = "Classification Accuracy")
lines!(ax_acc, 1:epochs+1, train_accuracy, color = :blue, label = "Training Accuracy")
lines!(ax_acc, 1:epochs+1, val_accuracy, color = :red, label = "Validation Accuracy")
hlines!(ax_acc, [test_accuracy], color = :green, linestyle = :dash, label = "Final Test Accuracy")
axislegend(ax_acc, backgroundcolor = :transparent, position = :rb)
# Add a title to the top of the figure
Label(
    fig[begin-1, 1:2],
    "Vanilla RNN, hidden size 100, 8,000 training examples, 250x250 matrices\nMeasurements: 50 traces of X * O_i with random O_i in R^{250x25}",
    fontsize = 20,
    padding = (0, 0, 0, 0),
)

fig
save("data/vanillaRNNwidth100nodynamics2.png", fig)