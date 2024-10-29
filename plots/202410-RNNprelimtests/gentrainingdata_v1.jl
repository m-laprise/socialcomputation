using Random
using JLD2

#include("genrandommatrix.jl")

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



####################

function trace_product(X::AbstractMatrix, O::AbstractMatrix)
    return sum(diag(X * O))
end

function trace_product(X::AbstractMatrix, O::Array)
    nb_measurements = size(O, 3)
    result = Vector{Float64}(undef, nb_measurements)
    temp_matrix = similar(X, size(X, 1), size(O, 2))
    diag_temp = Vector{Float64}(undef, min(size(X, 1), size(O, 2)))
    
    @inbounds for i in 1:nb_measurements
        @views mul!(temp_matrix, X, O[:, :, i])
        @inbounds for j in 1:length(diag_temp)
            diag_temp[j] = temp_matrix[j, j]
        end
        result[i] = sum(diag_temp)
    end
    return result
end

function get_data_densematrix(dataset_size::Int, m::Int, n::Int; 
        seed::Int, 
        measurement_type::String = "mask",
        mask_prob::Float64 = 0.33, 
        traces_nb::Int = 50,
        traces_projsize::Int = 25,
        include_X::Bool = false, 
        rank_threshold::Float64 = 0.1)

    cutoff = Int(round(min(m, n) * rank_threshold))
    maxrank = Int(round(min(m, n) / 2))

    rng = MersenneTwister(seed)
    # Generate matrix ranks
    indexablecollection = vcat(1:min(10,cutoff-2), max(maxrank-10, cutoff+2):maxrank)
    r = rand(rng, indexablecollection, dataset_size)
    n_seen = round(Int, (1 - mask_prob) * m * n)
    r = min.(r, fill(n_seen ÷ 2, dataset_size))

    # Preallocate the array for low-rank matrices
    X = Array{Float64, 3}(undef, dataset_size, m, n)
    # Generate (dataset_size) low-rank matrices from the rank vector
    for i in 1:dataset_size
        X[i, :, :] .= gen_matrix(m, n, r[i], seed=seed+i)
    end

    # Generate labels with shape (dataset_size, 1)
    Y = reshape(r .< cutoff, dataset_size, 1)

    if measurement_type == "mask"
        # Generate measurements of each X by masking some of the entries; dataset_size x m x n
        masks = rand(rng, Bool, dataset_size, m, n) .< (1 - mask_prob)
        Xhat = X .* masks
    elseif measurement_type == "trace"
        # Preallocate the array for trace measurements
        Xhat = Array{Float64, 2}(undef, dataset_size, traces_nb)
        # Generate measurements of each X by taking Trace(X * O_i) for (nb_traces) random matrices O_i
        O = randn(rng, dataset_size, n, traces_projsize, traces_nb)
        @inbounds for i in 1:dataset_size
            #Xhat[i, :] .= trace_product(X[i, :, :], O[i, :, :, :])
            X_i = view(X, i, :, :)
            for j in 1:traces_nb
                O_ij = view(O, i, :, :, j)
                Xhat[i, j] = trace_product(X_i, O_ij)
            end
        end
    else
        error("Measurement type not recognized. Choose from 'mask' or 'trace'.")
    end
    if include_X
        return X, Xhat, Y, r
    else
        return Xhat, Y, r
    end
end
