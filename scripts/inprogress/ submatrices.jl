include("genrandommatrix.jl")

m = 10000
n = 1000
r = 50

A = gen_matrix(m, n, r, noisy=true, noisevar=1e-1,sparse=false)
rank(A) == r

function submatrix(A, minval=0, maxval=0; seed=1234, type="partition")
    m, n = size(A)
    if m < n
        A = A'
        m, n = n, m
    end
    if minval == 0
        minval = 2 * rank(A)
    end
    if maxval == 0
        maxval = m/2
    end
    blocks = []
    if type == "partition"
        # Deterministically partition the matrix in blocks of min x min
        # and return all blocks
        for i in 1:minval:m
            for j in 1:minval:n
                push!(blocks, A[i:min(i+minval-1, m), j:min(j+minval-1, n)])
            end
        end
    elseif type == "random"
        k = m * n / (minval * minval)
        # Randomly sample a block of size min x min, k times
        rng = MersenneTwister(seed)
        for _ in 1:k
            i = rand(rng, 1:m-minval+1)
            j = rand(rng, 1:n-minval+1)
            push!(blocks, A[i:min(i+minval-1, m), j:min(j+minval-1, n)])
        end
    else 
        error("Invalid type")
    end
    return blocks
end

blocks_partition = submatrix(A, 50*2, 0, seed=1234, type="partition")
blocks_random = submatrix(A, 50*2, 0, seed=1234, type="random")

ranks_part = [rank(block) for block in blocks_partition]
ranks_rand = [rank(block) for block in blocks_random]

Ss = []
for block in blocks_partition
    _, S, _ = svd(block)
    S = sort(S, rev=true)
    S = log10.(S)
    push!(Ss, S)
end
Ss

using CairoMakie
CairoMakie.activate!(type = "png")

avgSs = []
stds = []
for i in 1:length(Ss[1])
    rankavg = mean([S[i] for S in Ss])
    rankstd = std([S[i] for S in Ss])
    push!(avgSs, rankavg)
    push!(stds, rankstd)
end


fig = Figure(size = (600, 600))
ax = Axis(fig[1, 1], title = "Spectrum of random submatrices", xlabel = "Singular value index", ylabel = "Singular value (log scale)")
#for S in Ss
#    line = lines!(ax, 1:length(Ss), Float64.(Ss), color = :blue, alpha = 0.1)
#end
lines!(ax, Float64.(avgSs) - Float64.(stds), color = :blue, alpha = 0.5, show_axis = false, show_grid = false)
lines!(ax, Float64.(avgSs) + Float64.(stds), color = :blue, alpha = 0.5, show_axis = false, show_grid = false)
lines!(ax, Float64.(avgSs), color = :red, show_axis = false, show_grid = false)
#for i in 1:length(avgSs)
 #   line = lines!(ax, [i, i], [avgSs[i] - stds[i], avgSs[i] + stds[i]], color = :blue)
#end
fig

