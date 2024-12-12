# Architectures that were tried and set aside

# To be tried next: 
# Cell with matrix-valued state;
# GRU, mGRU; 
# symmetric data; smooth data


# Main one
m_vanilla = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = false),
    dec = Dense(net_width => output_size)
)

# Works as well as main
m_vanilla = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = false),
    dec = Split(Dense(net_width => output_size), 
                Dense(net_width => output_size)),
    combine = x -> x[1] .+ x[2] 
)

# To try
m_vanilla = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = false),
    dec = Dense(net_width => output_size),
    sep = Split(x -> x[1:m], 
                x -> x[m+1:end]),
    combine = x -> vec(x[1] * x[2]')
)

# Works like random guessing
m_vanilla = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = false),
    dec = Split(Dense(net_width => m), 
                Dense(net_width => m)),
    combine = x -> vec(x[1] * x[2]') 
)

# So very slow, could not try locally
m_vanilla = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = false),
    dec = Flux.Bilinear(net_width => output_size)
)

# Vanilla with cob
# Same/slightly better performance than Vanilla
m_vanilla = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = false),
    dec = Dense(net_width => output_size),
    mat = x -> reshape(x, m, m),
    cob = BasisChange(m => m),
    flatten = x -> vec(x)
)

#====DUAL RNN CELL=====#
# Works as well as initial, but Jacobian is highly unstable (saved outputs)
m_vanilla = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = false, dual = true),
    combine = x -> x[:,1] .+ x[:,2], # also tried .*, did not learn
    dec = Dense(net_width => output_size)
)

# Works as well as initial, but Jacobian is highly unstable, forming a C around the unit circle. 
# Slower than above.
m_vanilla = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = false, dual = true),
    dec = Split(Chain(
                    pick = x -> x[:,1],
                    fwd = Dense(net_width => output_size)), 
                Chain(
                    pick = x -> x[:,2],
                    fwd = Dense(net_width => output_size))),
    combine = x -> x[1] .* x[2]  # Also tried .+
)
# When combining before the feed forward layer, only adding the duals work.
# When combining after a dual feed forward (slower), both adding and multiplying the duals work.

# Next I tried to add a change of basis layer after the feed forward layer.
# It reaches same RMSE. but lower nnm-penalized loss:
# Epoch 20: Train loss: 0.1074; train RMSE: 0.9354
#           Test loss: 0.1116; Test RMSE: 0.9358
# Jacobian looks different, but still unstable. Saved as "dualwithcob"
m_vanilla = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = false, dual = true),
    combine = x -> x[:,1] .+ x[:,2],
    dec = Dense(net_width => output_size),
    mat = x -> reshape(x, m, m),
    cob = BasisChange(m => m),
    flatten = x -> vec(x)
)

# Not better than random guessing (no feed forward)
m_vanilla = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = false, dual = true),
    combine = x -> (x[:,1] * x[:,2]'),
    dec = BasisChange(net_width => m),
    flatten = x -> vec(x)
)

# still not better than random guessing (with feed forward)
m_vanilla = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = false, dual = true),
    ffwd = Dense(net_width => net_width),
    combine = x -> (x[:,1] * x[:,2]'),
    dec = BasisChange(net_width => m),
    flatten = x -> vec(x)
)
