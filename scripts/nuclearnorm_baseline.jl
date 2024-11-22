import Pkg
Pkg.activate("../")

using LinearAlgebra
using Random
using Distributions
using DataFrames
using Statistics
using Dates
using CSV
using CairoMakie

include("sota_matrix_completion.jl")
using InteractiveUtils: versioninfo
println(versioninfo(verbose=true))

# get NUM from the command line
if length(ARGS) == 0
    println("Error: NUM argument missing.")
    exit(1)
end
NUM = parse(Int, ARGS[1])
if NUM < 1 || NUM > 9
    println("Error: NUM must be an integer between 1 and 9.")
    exit(1)
end

function nnm_setup(n, r, alpha, seed)
    rng = Random.MersenneTwister(seed)
    A = randn(rng, n, r) * randn(rng, r, n)
    B = A .* (rand(rng, n, n) .< alpha)
    mask = B .!= 0
    return A, B, mask
end
MSE(A::AbstractArray, B::AbstractArray) = mean((A .- B).^2)
RMSE(A::AbstractArray, B::AbstractArray) = sqrt(MSE(A, B))

function run_nnm(A, B, mask, f)
    ans = @timed f(B, mask)
    return (ans.time, RMSE(ans.value, A))
end

function nnm_test(k, n, r, alpha, initial_seed; doPYnnm=false, doSCSnnm=true)
    @assert doPYnnm || doSCSnnm
    if doPYnnm
        pyresults = []
    end
    if doSCSnnm
        scsresults = []
    end
    for i in 1:k
        seed = initial_seed + i
        A, B, mask = nnm_setup(n, r, alpha, seed)
        if doPYnnm
            push!(pyresults, run_nnm(A, B, mask, PYnnm))
        end
        if doSCSnnm
            push!(scsresults, run_nnm(A, B, mask, SCSnnm))
        end
    end
    if doPYnnm && doSCSnnm
        return pyresults, scsresults
    elseif doPYnnm
        return pyresults
    else
        return scsresults
    end
end

function get_stats(results; tol = 1e-4)
    times = [x[1] for x in results]
    rmses = [x[2] for x in results]
    return (mean_time=mean(times), std_time=std(times), 
            min_time=minimum(times), max_time=maximum(times), median_time=median(times),
            mean_rmse=mean(rmses), std_rmse=std(rmses), 
            min_rmse=minimum(rmses), max_rmse=maximum(rmses), median_rmse=median(rmses),
            pct_success_rmse=sum([x < tol for x in rmses]) / length(rmses))
end

function record_stats(k, n, r, alpha, initial_seed; doPYnnm=false, doSCSnnm=true)
    @assert doPYnnm || doSCSnnm
    if doPYnnm && doSCSnnm
        pyres, scsres = nnm_test(k, n, r, alpha, initial_seed; doPYnnm=true, doSCSnnm=true)
        pystats = get_stats(pyres)
        scsstats = get_stats(scsres)
        return DataFrame(
            k=k, n=n, r=r, alpha=alpha, initial_seed=initial_seed,
            method=["PYnnm","PYnnm","SCSnnm","SCSnnm"],
            metric=["time","RMSE","time","RMSE"],
            mean=[pystats.mean_time, pystats.mean_rmse, scsstats.mean_time, scsstats.mean_rmse],
            std=[pystats.std_time, pystats.std_rmse, scsstats.std_time, scsstats.std_rmse],
            min=[pystats.min_time, pystats.min_rmse, scsstats.min_time, scsstats.min_rmse],
            median=[pystats.median_time, pystats.median_rmse, scsstats.median_time, scsstats.median_rmse],
            max=[pystats.max_time, pystats.max_rmse, scsstats.max_time, scsstats.max_rmse],
            pct_success=[pystats.pct_success_rmse, pystats.pct_success_rmse, 
                         scsstats.pct_success_rmse, scsstats.pct_success_rmse])
    else 
        if doPYnnm
            pyres = nnm_test(k, n, r, alpha, initial_seed; doPYnnm=true, doSCSnnm=false)
            resstats = get_stats(pyres)
            curmethod = "PYnnm"
        else
            scsres = nnm_test(k, n, r, alpha, initial_seed; doPYnnm=false, doSCSnnm=true)
            resstats = get_stats(scsres)
            curmethod = "SCSnnm"
        end
        return DataFrame(
            k=k, n=n, r=r, alpha=alpha, initial_seed=initial_seed,
            method=curmethod,
            metric=["time","RMSE"],
            mean=[resstats.mean_time, resstats.mean_rmse],
            std=[resstats.std_time, resstats.std_rmse],
            min=[resstats.min_time, resstats.min_rmse],
            median=[resstats.median_time, resstats.median_rmse],
            max=[resstats.max_time, resstats.max_rmse],
            pct_success=[resstats.pct_success_rmse, resstats.pct_success_rmse])
    end
end

# RUN TESTS

n_range = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
alpha_range = [i for i in 0.05:0.05:0.95]
# h(n) = n < 100 ? 1 : n / 100
# derive_r_range(n) = sort(unique([Int(round(n/i)) for i in 2:h(n):n])) # local
derive_r_range(n) = [i for i in 1:min(div(n, 4), 30)] # on HPC

df = DataFrame(
    k=Int[], n=Int[], r=Int[], alpha=Float64[], initial_seed=Int[],
    method=String[], metric=String[], mean=Float64[], std=Float64[], 
    min=Float64[], median=Float64[], max=Float64[], pct_success=Float64[])

for n in n_range[NUM]
    k = n < 100 ? 100 : 20  
    for alpha in alpha_range
        for r in derive_r_range(n)
            data = record_stats(k, n, r, alpha, 192743)
            append!(df, data)
            println("Finished r = ", r, " at ", Dates.format(now(), "HH:MM:SS"))
        end
        println("Finished alpha = ", alpha, " at ", Dates.format(now(), "HH:MM:SS"))
    end
    println("Finished n = ", n, " at ", Dates.format(now(), "HH:MM:SS"))
end

# Write df to csv
# make sure directory exists; if not, create it
if !isdir("output")
    mkdir("output")
end
CSV.write("output/baseline_nnm_results.csv", df, append=true)

# For a given matrix size, create three heatmap plots, each with r on the x-axis and alpha on the y-axis. 
# The three plots should show the mean time, mean RMSE, and the percentage of success.
f = Figure(size=(1200, 400))
titles = ["Mean Time (in seconds)", "Mean RMSE", "Proportion of Success (tol = 1e-4)"]
selectedcols = [:mean, :mean, :pct_success]
metricfilters = ["time", "RMSE", "RMSE"]
for i in 1:3
    plotdata = filter(row -> row.metric == metricfilters[i], df)
    ax, hm = contourf(
        f[1, i][1,1], plotdata[!, :r], plotdata[!, :alpha], plotdata[!, selectedcols[i]], 
        visible=true)
    Colorbar(f[:, i][1,2], hm, vertical = true)
    ax.title = titles[i]
    ax.xlabel = "Rank"
    ax.ylabel = "Proportion of known entries"
end
Label(f[0, :], text = "Performance of low-rank matrix completion via nuclear norm minimization\n"*
                      "over $(df[1, :k]) random n x n matrices of size n = $(df[1, :n])", 
      fontsize = 24)
save("output/baseline_nnm_results_$(df[1, :n]).png", f)
