using LinearAlgebra
using Random
using Distributions
using DataFrames
using Statistics
using Dates
using CSV
using CairoMakie

include("sota_matrix_completion.jl")

NUM = 1
fNUM = 1

function setup(m, n, r, alpha, seed)
    rng = Random.MersenneTwister(seed)
    A = (randn(rng, m, r) ./ sqrt(sqrt(r))) * (randn(rng, r, n) ./ sqrt(sqrt(r)))
    B = A .* (rand(rng, m, n) .< alpha)
    mask = B .!= 0
    return A, B, mask
end
MSE(A::AbstractArray, B::AbstractArray) = abs.(mean((A .- B).^2))
RMSE(A::AbstractArray, B::AbstractArray) = abs.(sqrt(MSE(A, B)))
spectraldist(A, B) = norm(abs.(svdvals(A)) - abs.(svdvals(B))) / length(svdvals(A))

function run_algo(A, B, mask, r, f; kwargs...)
    m, n = size(B)
    I_idx, J_idx, knownentries = sparse2idx(B)
    ans = @timed f(m, n, I_idx, J_idx, knownentries, r; kwargs...)
    # RMSE of unknown entries only, RMSE of all entries, spectral distance
    return (ans.time, 
            RMSE(ans.value[.~mask], A[.~mask]), 
            RMSE(ans.value, A),
            spectraldist(ans.value, A))
end

function algo_test(k, m, n, r, alpha, initial_seed, f; kwargs...)
    algoresults = Vector{Any}(undef, k)
    for i in 1:k
        seed = initial_seed + i
        A, B, mask = setup(m, n, r, alpha, seed)
        algoresults[i] = run_algo(A, B, mask, r, f; kwargs...)
    end
    return algoresults
end

function calculate_stats(data)
    return (
        mean=mean(data), std=std(data), min=minimum(data), 
        max=maximum(data), median=median(data)
    )
end

function get_stats(results; tol = 1e-4)
    times = [x[1] for x in results]
    rmses = [x[2] for x in results]
    rmses_all = [x[3] for x in results]
    spectraldists = [x[4] for x in results]
    time_stats = calculate_stats(times)
    rmse_stats = calculate_stats(rmses)
    rmse_all_stats = calculate_stats(rmses_all)
    specdist_stats = calculate_stats(spectraldists)
    pct_success_rmse = sum([x < tol for x in rmses_all]) / length(rmses_all)
    return merge(
        (mean_time=time_stats.mean, std_time=time_stats.std, min_time=time_stats.min, median_time=time_stats.median, max_time=time_stats.max,),
        (mean_rmse=rmse_stats.mean, std_rmse=rmse_stats.std, min_rmse=rmse_stats.min, median_rmse=rmse_stats.median, max_rmse=rmse_stats.max,),
        (mean_rmse_all=rmse_all_stats.mean, std_rmse_all=rmse_all_stats.std, min_rmse_all=rmse_all_stats.min, median_rmse_all=rmse_all_stats.median, max_rmse_all=rmse_all_stats.max,),
        (mean_specdist=specdist_stats.mean, std_specdist=specdist_stats.std, min_specdist=specdist_stats.min, median_specdist=specdist_stats.median, max_specdist=specdist_stats.max,),
        (pct_success_rmse=pct_success_rmse,))
end

function record_stats(k, m, n, r, alpha, initial_seed, f; kwargs...)
    algores = algo_test(k, m, n, r, alpha, initial_seed, f; kwargs...)
    resstats = get_stats(algores)
    return DataFrame(
        k=k, m=m, n=n, r=r, alpha=alpha, initial_seed=initial_seed,
        method=string(nameof(f)),
        metric=["time","RMSE_unknown","RMSE_all","spectraldist"],
        mean=[resstats.mean_time, resstats.mean_rmse, resstats.mean_rmse_all, resstats.mean_specdist],
        std=[resstats.std_time, resstats.std_rmse, resstats.std_rmse_all, resstats.std_specdist],
        min=[resstats.min_time, resstats.min_rmse, resstats.min_rmse_all, resstats.min_specdist],
        median=[resstats.median_time, resstats.median_rmse, resstats.median_rmse_all, resstats.median_specdist],
        max=[resstats.max_time, resstats.max_rmse, resstats.max_rmse_all, resstats.max_specdist],
        pct_success=[NaN, NaN, resstats.pct_success_rmse, NaN])
end

#n_range = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
#alpha_range = [i for i in 0.05:0.1:0.95] #local
#alpha_range = [i for i in 0.05:0.05:0.95] #HPC
alpha_range = [i for i in 0.01:0.01:0.1] 
n_range = [100]
#h(n) = n < 100 ? 1 : n / 100
derive_r_range(n) = [i for i in 1:4]
#derive_r_range(n) = sort(unique([Int(round(n/i)) for i in 2:h(n):n])) # local
#derive_r_range(n) = [i for i in 1:min(div(n, 2), 30)] # on HPC

myfunctions = [ScaledASD, IRLS_M]
mykwargs = Dict(:IRLS_M => Dict(:q => 8, :alpha => 1.0), 
                :ScaledASD => Dict(:opts => Dict(
                    :rel_res_tol => 1e-7,
                    :maxit => 10000,
                    :verbosity => true,
                    :rel_res_change_tol => 1e-5)))

df = DataFrame(
    k=Int[], m=Int[], n=Int[], r=Int[], alpha=Float64[], initial_seed=Int[],
    method=String[], metric=String[], mean=Float64[], std=Float64[], 
    min=Float64[], median=Float64[], max=Float64[], pct_success=Float64[])

activefunction = myfunctions[fNUM]
activekwargs = mykwargs[nameof(activefunction)]

# Run once to force compilation before timing
println("Compiling functions...")
println(record_stats(20, 80, 80, 1, 0.108, 147, activefunction; activekwargs...))

println("Starting loop... at ", Dates.format(now(), "HH:MM:SS"))
for n in n_range[NUM]
    k = n < 100 ? 100 : 20
    for alpha in alpha_range
        for r in derive_r_range(n)
            data = record_stats(k, n, n, r, alpha, 93759, activefunction; activekwargs...)
            append!(df, data)
            println("Finished r = ", r, " at ", Dates.format(now(), "HH:MM:SS"))
            #println("========================================================")
        end
        println("Finished alpha = ", alpha, " at ", Dates.format(now(), "HH:MM:SS"))
        println("========================================================")
        println("========================================================")
    end
    println("Finished n = ", n, " at ", Dates.format(now(), "HH:MM:SS"))
end

# Write df to csv
# make sure directory exists; if not, create it
subfolder = "data"
if !isdir(subfolder)
    mkdir(subfolder)
end
CSV.write("$(subfolder)/baseline_$(string(nameof(activefunction)))_results.csv", df, append=true)

# For a given matrix size, create three heatmap plots, each with r on the x-axis and alpha on the y-axis. 
# The three plots should show the mean time, mean RMSE, and the percentage of success.
df2 = filter(row -> row.r != 64, df)
f = Figure(size=(1200, 700))
titles = ["Mean time (in log seconds)", "O(mean RMSE) for unobserved entries", 
            "Proportion of success (tol = 1e-4)"]
selectedcols = [:mean, :mean, :pct_success]
metricfilters = ["time", "RMSE_unknown", "RMSE_all"]
for i in 1:3
    plotdata = filter(row -> row.metric == metricfilters[i], df2)
    zvalues = i == 2 ? log10.(plotdata[!, selectedcols[i]]) : plotdata[!, selectedcols[i]]
    if i == 1
        ax, hm = contourf(
            f[1, i][1,1], plotdata[!, :r], plotdata[!, :alpha], log10.(zvalues), colormap = :viridis,
            visible=true)
        Colorbar(f[:, i][1,2], hm, vertical = true)
    elseif i == 2
        zrange = Int(floor(maximum(zvalues)) - floor(minimum(zvalues)))
        ax, hm = contourf(
            f[1, i][1,1], plotdata[!, :r], plotdata[!, :alpha], zvalues, colormap = :viridis, levels=zrange,
            visible=true)
        Colorbar(f[:, i][1,2], hm, vertical = true,
                 tickformat = values -> ["1e$(round(value, digits=0))" for value in values])
    else
        ax, hm = contourf(
            f[1, i][1,1], plotdata[!, :r], plotdata[!, :alpha], zvalues, colormap = :viridis,
            visible=true)
        Colorbar(f[:, i][1,2], hm, vertical = true)
    end
    ax.title = titles[i]
    ax.xlabel = "Rank"
    ax.xticks = 0:1:maximum(plotdata[!, :r])
    ax.ylabel = "Proportion of known entries"
end

plot5data = filter(row -> row.metric == "RMSE_all", df2)
ax5, hm5 = contourf(
            f[2, 2][1,1], plot5data[!, :r], plot5data[!, :alpha], log10.(plot5data[!, :mean]), colormap = :viridis,
            visible=true)
Colorbar(f[2, 2][1,2], hm5, vertical = true,
         tickformat = values -> ["1e$(round(value, digits=0))" for value in values])
ax5.title = "O(mean RMSE) for all entries"
ax5.xlabel = "Rank"
ax5.xticks = 0:1:maximum(plot5data[!, :r])
ax5.ylabel = "Proportion of known entries"

plot6data = filter(row -> row.metric == "spectraldist", df2)
ax6, hm6 = contourf(
            f[2, 3][1,1], plot6data[!, :r], plot6data[!, :alpha], log10.(plot6data[!, :mean]), colormap = :viridis,
            visible=true)
Colorbar(f[2, 3][1,2], hm6, vertical = true,
         tickformat = values -> ["1e$(round(value, digits=0))" for value in values])
ax6.title = "O(mean spectral distance)"
ax6.xlabel = "Rank"
ax6.xticks = 0:1:maximum(plot6data[!, :r])
ax6.ylabel = "Proportion of known entries"

Label(f[0, :], text = "Performance of low-rank matrix completion via $(nameof(activefunction)) algorithm\n"*
                      "over $(df2[1, :k]) random n x n matrices of size n = $(df2[1, :n])", 
      fontsize = 24)
f

save("$(subfolder)/baseline_$(nameof(activefunction))_results_$(df2[1, :n]).png", f)
