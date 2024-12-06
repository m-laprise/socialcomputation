# Architectures that were tried and set aside

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

# 
m_vanilla = Chain(
    rnn = rnn(input_size, net_width;
        Whh_init = Whh_init, 
        h_init = "randn",
        gated = false),
    proj = x -> diagm(0 => x),
    # ... how to go from here?
)

#(F) dec = Dense(transpose(rnn.Whh))
