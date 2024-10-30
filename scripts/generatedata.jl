#= 
This script generates the data for the first round of RNN experiments,
which includes data for the following tasks:
- Classification tasks with low-rank matrices (lr_c1, lr_c2)
    - lr_c1: ranks 1-8, 121-128, matrix size 256 (binary classification, high vs low rank)
    - lr_c2: ranks 2, 4, 8, 16, 32, 64, 128, 256, matrix size 256 (multi-class classification)
- Reconstruction tasks with low-rank matrices (lr_r_32, lr_r_64, lr_r_128, lr_r_256)
    - rank 8, matrix sizes 32, 64, 128, 256
- Reconstruction tasks with sparse matrices (sparse_200)
    - rank 2, sparsity 1/3, matrix size 200
    
The data is saved in a JLD2 file with Bzip2 compression.
=#

using Random
using JLD2, CodecBzip2

include("genrandommatrix.jl")

dataset_size = 10000
seed = 5793

# LOW-RANK MATRICES 
# Classification tasks

m = 256

r = [1:8; 121:128]
lr_c1_X, lr_c1_ranks = generate_matrix_set(m, m, dataset_size, r; seed=seed)
seed += 1

#= size(lr_c1_X)
Base.summarysize(lr_c1_X) / 1024^2
lr_c1_X_list = [lr_c1_X[:, :, i] for i in 1:dataset_size]
Base.summarysize(lr_c1_X_list) / 1024^2 =#

r = [2,4,8,16,32,64,128,256]
lr_c2_X, lr_c2_ranks = generate_matrix_set(m, m, dataset_size, r; seed=seed)
seed += 1

lr_c1 = Dict(
    "X" => lr_c1_X,
    "ranks" => lr_c1_ranks
)

lr_c2 = Dict(
    "X" => lr_c2_X,
    "ranks" => lr_c2_ranks
)

datafilepath = "data/rnn_firstexpdata.jld2"
jldsave(datafilepath, compress=Bzip2Compressor();
        lr_c1, lr_c2)

lr_c1_X = nothing
lr_c1_ranks = nothing
lr_c2_X = nothing
lr_c2_ranks = nothing
lr_c1 = nothing
lr_c2 = nothing

# Reconstruction task
matrix_sizes = [32, 64, 128, 256] #, 1024, 2048, 4096]
r = [8]
for m in matrix_sizes
    obj_X, obj_ranks = generate_matrix_set(m, m, dataset_size, r; seed=seed)
    seed += 1
    println("Generated dataset for matrix size $m.")
    println("Memory usage: $(Base.summarysize(obj_X) / 1024^2) MB")
    jldopen(datafilepath, "a"; compress=Bzip2Compressor()) do f
            f["lr_r_$(m)"] = Dict("X" => obj_X, "ranks" => obj_ranks)
    println("Saved to file.")
    end
    obj_X = nothing
    obj_ranks = nothing
end

# SPARSE MATRICES
r = [2]
m = 200
sparse_X, sparse_ranks = generate_matrix_set(m, m, dataset_size, r; sparsity=1/3, seed=seed)
seed += 1
jldopen(datafilepath, "a"; compress=Bzip2Compressor()) do f
    f["sparse_$(m)"] = Dict("X" => sparse_X, "ranks" => sparse_ranks)
end

sparse_X = nothing
sparse_ranks = nothing


# MEASUREMENTS

# Traces of projections
seed = 90541
nbtraces = 50
projsize = 50

datasetnames = ["lr_c1", "lr_c2", "lr_r_32", "lr_r_64", "lr_r_128", "lr_r_256"]

function take_traces(datasetname; nbtraces, projsize, seed=Int(round(time())))
    data = load("data/rnn_firstexpdata.jld2", datasetname)
    _, n, dataset_size = size(data["X"])
    rng = MersenneTwister(seed)
    O = randn(rng, Float32, n, projsize, nbtraces, dataset_size)
    Xhat = trace_measurements(data["X"], O)
    return Xhat
end

tracesfilepath = "data/rnn_firstexptraces.jld2"

for datasetname in datasetnames
    Xhat = take_traces(datasetname; nbtraces=nbtraces, projsize=projsize, seed=seed)
    seed += 1
    println("Generated traces for dataset $datasetname.")
    jldopen(tracesfilepath, "a"; compress=Bzip2Compressor()) do f
        f["$(datasetname)_traces"] = Xhat
    end
    println("Saved to file.")
    Xhat = nothing
end
