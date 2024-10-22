using Flux
using Flux.Losses: mse

device = Flux.get_device(; verbose=true)

#############

# Dense layer that acts on two inputs, h and x.
# If you run the last line a few times, you'll notice the output y changing slightly even though the input x is the same.

output_size = 5
input_size = 2
Wxh = randn(Float32, output_size, input_size)
Whh = randn(Float32, output_size, output_size)
b   = randn(Float32, output_size)

# This is the same as rnn_cell = Flux.RNNCell(2, 5) but defined manually
function rnn_cell(h, x)
    h = tanh.(Wxh * x .+ Whh * h .+ b)
    return h, h
end

x = rand(Float32, input_size) # dummy input data
h = rand(Float32, output_size) # random initial hidden state

# STATELESS RNN (h is handled manually)
h, y = rnn_cell(h, x)

# STATEFUL RNN
# The Recur wrapper stores the state between runs in the m.state field
# This is similar to the RNN(2, 5) constructor--or RNN(2 => 5)--an RNNCell wrapped in Recur.
m = Flux.Recur(rnn_cell, h)
y = m(x)

# Complete network with feedforward final layer
m = Chain(
    #Flux.Recur(rnn_cell, h),
    RNN(input_size => output_size),
    Dense(output_size => 1)
)
y = m(x)
m.layers[1].state

# apply the model to a sequence of inputs

# The m(x) operation would be represented by x1 -> A -> y1 in a diagram
# If we perform this operation a second time, it will be equivalent to x2 -> A -> y2 
# since the model m has stored the state resulting from the x1 step.

# instead of computing a single step at a time, we can get the full y1 to y3 sequence 
# in a single pass by iterating the model on a sequence of data.

# To do so, we structure the input data as a Vector of observations at each time step. 
# This Vector is of length = seq_length and each of its elements represent the input features for a given step. 
# In our example, this translates into a Vector of length 3, where each element is a Matrix of size (features, batch_size), 
# or just a Vector of length features if dealing with a single observation.
x = [rand(Float32, 2) for i = 1:3];
[m(xi) for xi in x]

# TRAINING:
# SEQ-TO-ONE structure

# Exclude the first step of the RNN chain for the computation of the loss
# Only the last two outputs are used to compute the loss, hence the target y being of length 2. 
# This strategy can easily handle a seq-to-one structure

function loss(x, y)
  m(x[1]) # ignores the output but updates the hidden states
  sum(mse(m(xi), yi) for (xi, yi) in zip(x[2:end], y))
end
y = [rand(Float32, 1) for i=1:2]
loss(x, y)

# Warm-up of the hidden state followed with a regular training where 
# all the steps of the sequence are considered for the gradient update
function loss(m, x, y)
  sum(mse(m(xi), yi) for (xi, yi) in zip(x, y))
end
# Here, a single continuous sequence is considered.
# The model state is not reset between the 2 batches, so the state of the model flows through the batches,
# which only makes sense in the context where seq_1 is the continuation of seq_init and so on.

seq_init = [rand(Float32, 2)]
seq_1 = [rand(Float32, 2) for i = 1:3]
seq_2 = [rand(Float32, 2) for i = 1:3]

y1 = [rand(Float32, 1) for i = 1:3]
y2 = [rand(Float32, 1) for i = 1:3]

X = [seq_1, seq_2]
Y = [y1, y2]
data = zip(X,Y)

# model's state is first reset
Flux.reset!(m)
# a warmup is performed over a sequence of length 1 
# by feeding it with seq_init, resulting in a warmup state
[m(x) for x in seq_init]

# model is trained for 1 epoch, where 2 batches are provided (seq_1 and seq_2) 
# and all the timesteps outputs are considered for the loss.
opt = Flux.setup(Adam(1e-3), m)
Flux.train!(loss, m, data, opt)
# examine hidden state
m.layers[1].state
# examine trained weights and biases
Flux.params(m)[1]
Flux.params(m)[2]
Flux.params(m)[3]
Flux.params(m)[4]
Flux.params(m)[5]
Flux.params(m)[6]

# Batch size is 1 here as there's only a single sequence within each batch. 
# If the model was to be trained on multiple independent sequences, 
# these sequences could be added to the input data as a second dimension. 

# For example, in a language model, each batch would contain multiple independent sentences. 
# In such scenario, if we set the batch size to 4, a single batch would be of the shape:
x = [rand(Float32, 2, 4) for i = 1:3]
y = [rand(Float32, 1, 4) for i = 1:3]
# That would mean that we have 4 sentences (or samples), each with 2 features (let's say a very small embedding!) 
# and each with a length of 3 (3 words per sentence). 
# Computing m(batch[1]), would still represent x1 -> y1 in our diagram and returns the first word output, 
# but now for each of the 4 independent sentences (second dimension of the input matrix). 
# We do not need to use Flux.reset!(m) here; 
# each sentence in the batch will output in its own "column", and the outputs of the different sentences won't mix.

# EXAMPLE OF BATCHING
output_size = 5
input_size = 2
Wxh = randn(Float32, output_size, input_size)
Whh = randn(Float32, output_size, output_size)
b   = randn(Float32, output_size)

function rnn_cell(h, x)
    h = tanh.(Wxh * x .+ Whh * h .+ b)
    return h, h
end
# We use the last dimension of the input and the hidden state as the batch dimension. 
# I.e., h[:, n] would be the hidden state of the nth sentence in the batch.

batch_size = 4
x = rand(Float32, input_size, batch_size) # dummy input data
h = rand(Float32, output_size, batch_size) # random initial hidden state

h, y = rnn_cell(h, x)

size(h) == size(y) == (output_size, batch_size)

# In many situations, eg language model, the sentences in each batch are independent 
# (i.e. the last item of the first sentence of the first batch is independent from 
# the first item of the first sentence of the second batch), so we cannot handle the model as if 
# each batch was the direct continuation of the previous one. 
# To handle such situations, we need to reset the state of the model between each batch, 
# which can be conveniently performed within the loss function:
function loss(x, y)
  Flux.reset!(m)
  sum(mse(m(xi), yi) for (xi, yi) in zip(x, y))
end

# A potential source of ambiguity with RNN in Flux can come from the different data layout 
# compared to some common frameworks where data is typically a 3 dimensional array: 
# (features, seq length, samples). 
# In Flux, those 3 dimensions are provided through a vector of seq length containing a matrix (features, samples).



#################
# Language modeling with teacher forcing example

function loss(model, xs, ys)
    Flux.reset!(model)
    return sum(Flux.logitcrossentropy.([model(x) for x in xs], ys))
end

function predict(model::Chain, prefix::String, num_preds::Int)
    model = cpu(model)
    Flux.reset!(model)
    buf = IOBuffer()
    write(buf, prefix)

    c = wsample(vocab, softmax([model(Flux.onehot(c, vocab)) for c in collect(prefix)][end]))
    for i in 1:num_preds
        write(buf, c)
        c = wsample(vocab, softmax(model(Flux.onehot(c, vocab))))
    end
    return String(take!(buf))
end

vocab_len = 1024
model = Chain(
    GRUv3(vocab_len => 32),
    Dense(32 => vocab_len)
) |> device
opt_state = Flux.setup(Adam(1e-2), model)

loss_train = []

for epoch in 1:50
    Flux.reset!(model)
    Flux.train!(loss,model,data,opt_state)
    push!(loss_train,sum(loss.(Ref(model), x, y)) / length(str)) 
end

using CairoMakie
fg,ax = lines(loss_train; axis=(;xlabel = "epoch",ylabel = "loss"))