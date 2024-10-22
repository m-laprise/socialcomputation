include("genrandommatrix.jl")
using CairoMakie
CairoMakie.activate!(type = "png")

m = 10000
n = 1000
r = 50

A = gen_matrix(m, n, r, noisy=true, noisevar=1e-1,sparse=false)
rank(A) == r

G = gen_matrix(60, 40, 5, noisy=false, sparse=false, spatial=true)

u, s, vt = svd(G)
# Approximate G from first 2 singular values and vectors
num = 1
Ghat = u[:, 1:num] * diagm(0 => s[1:num]) * vt[:, 1:num]'

# Visualize G as a heatmap
fig = Figure(size = (600, 400))
ax = Axis(
    fig[1, 1], title = "Matrix with field structure, r=$(rank(Ghat))",
    xlabel = "Column index", ylabel = "Row index")
heatmap!(ax, Ghat, colormap = :viridis)
fig




blocks_partition = submatrix(A, 50*2, 0, seed=1234, type="partition")
blocks_random = submatrix(A, 50*2, 0, seed=1234, type="random")

ranks_part = [rank(block) for block in blocks_partition]
ranks_rand = [rank(block) for block in blocks_random]

Ss = []
for block in blocks_partition
    _, S, _ = svd(block)
    S = sort(S, rev=true)
    S = log.(S)
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
for S in Ss
    line = lines!(ax, 1:length(S), S, color = :gray, alpha = 0.05)
end
#lines!(ax, Float64.(avgSs) - Float64.(stds), color = :blue, alpha = 0.5, show_axis = false, show_grid = false)
#lines!(ax, Float64.(avgSs) + Float64.(stds), color = :blue, alpha = 0.5, show_axis = false, show_grid = false)
lines!(ax, Float64.(avgSs), color = :red, show_axis = false, show_grid = false)
#for i in 1:length(avgSs)
 #   line = lines!(ax, [i, i], [avgSs[i] - stds[i], avgSs[i] + stds[i]], color = :blue)
#end
fig

