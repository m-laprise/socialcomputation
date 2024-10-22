using Random
using JLD2

include("genrandommatrix.jl")

dataset_size = 10000
m = 250
n = 250
seed = 8958

# For memory issues, generate in 4 chunks of dataset_size/4 and then concatenate
chunk_nb = 10
chunk_size = Int(dataset_size / chunk_nb)
X = nothing
Xmask = nothing
Xtrace = nothing
Y = nothing
ranks = nothing
for i in 1:chunk_nb
    println("Chunk $i")
    X_chunk, Xmask_chunk, Y_chunk, ranks_chunk = get_data_densematrix(
        chunk_size, m, n, seed=seed+i,
        measurement_type="mask", 
        mask_prob = 0.5,
        rank_threshold = 0.25,
        include_X=true)
    println("Mask generated")
    Xtrace_chunk, _, _ = get_data_densematrix(
        chunk_size, m, n, seed=seed+i,
        measurement_type="trace", 
        traces_nb = 50,
        traces_projsize = 25,
        rank_threshold = 0.25,
        include_X=false)
    println("Trace generated")
    if i == 1
        X = X_chunk
        Xmask = Xmask_chunk
        Xtrace = Xtrace_chunk
        Y = Y_chunk
        ranks = ranks_chunk
    else
        X = vcat(X, X_chunk)
        Xmask = vcat(Xmask, Xmask_chunk)
        Xtrace = vcat(Xtrace, Xtrace_chunk)
        Y = vcat(Y, Y_chunk)
        ranks = vcat(ranks, ranks_chunk)
    end
end
    
mean(Y)
# Convert to Float32
Xmask = Float32.(Xmask)
Xtrace = Float32.(Xtrace)
Y = Float32.(Y)

size(X), size(Xmask), size(Xtrace), size(Y), size(ranks)

# Save training data for re-use
save("data/rnn_flux_data.jld2","X", X, "Xmask", Xmask, "Xtrace", Xtrace, "Y", Y, "ranks", ranks)