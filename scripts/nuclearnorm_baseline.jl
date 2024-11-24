#import Pkg
#Pkg.activate(".")

using LinearAlgebra
using Random
using Distributions
using DataFrames
using Statistics
using Dates
using CSV
using CairoMakie

include("sota_matrix_completion.jl")
#using InteractiveUtils: versioninfo
#println(versioninfo(verbose=true))

# get NUM from the command line
#=
if length(ARGS) == 0
    println("Error: NUM argument missing.")
    exit(1)
end
NUM = parse(Int, ARGS[1])
if NUM < 1 || NUM > 9
    println("Error: NUM must be an integer between 1 and 9.")
    exit(1)
end
=#

function nnm_setup(m, n, r, alpha, seed)
    rng = Random.MersenneTwister(seed)
    A = (randn(rng, m, r) ./ sqrt(sqrt(r))) * (randn(rng, r, n) ./ sqrt(sqrt(r)))
    B = A .* (rand(rng, m, n) .< alpha)
    mask = B .!= 0
    return A, B, mask
end
MSE(A::AbstractArray, B::AbstractArray) = mean((A .- B).^2)
RMSE(A::AbstractArray, B::AbstractArray) = sqrt(MSE(A, B))

function run_nnm(A, B, mask, f)
    ans = @timed f(B, mask)
    # RMSE of unknown entries only
    return (ans.time, RMSE(ans.value[.~mask], A[.~mask]))
end

#=
using Hypatia
function hypatnnm(A, mask; verbose=false)
    model = Model(Hypatia.Optimizer)
    #MOI.set(model, MOI.RawOptimizerAttribute("eps_abs"), 1e-5)
    #MOI.set(model, MOI.RawOptimizerAttribute("eps_rel"), 1e-5)
    n = size(A, 1)
    @variable(model, X[1:n, 1:n])
    @constraint(model, X[mask] .== A[mask])
    @variable(model, t)
    @constraint(model, [t; vec(X)] in MOI.NormNuclearCone(n, n))
    @objective(model, Min, t)
    if !verbose
        set_silent(model)
    end
    optimize!(model)
    return value.(X)
end

n = 30
r = 2
alpha = 0.3
seed = 392743
A, B, mask = nnm_setup(n, r, alpha, seed)
scsresults = run_nnm(A, B, mask, SCSnnm)
hypres = run_nnm(A, B, mask, hypatnnm)
@benchmark SCSnnm($B, $mask)
@benchmark hypatnnm($B, $mask)
=#

function nnm_test(k, m, n, r, alpha, initial_seed; doPYnnm=false, doSCSnnm=true)
    @assert doPYnnm || doSCSnnm
    if doPYnnm
        pyresults = []
        sizehint!(pyresults, k)
    end
    if doSCSnnm
        scsresults = []
        sizehint!(scsresults, k)
    end
    for i in 1:k
        seed = initial_seed + i
        A, B, mask = nnm_setup(m, n, r, alpha, seed)
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

function record_stats(k, m, n, r, alpha, initial_seed; doPYnnm=false, doSCSnnm=true)
    @assert doPYnnm || doSCSnnm
    if doPYnnm && doSCSnnm
        pyres, scsres = nnm_test(k, m, n, r, alpha, initial_seed; doPYnnm=true, doSCSnnm=true)
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
            pyres = nnm_test(k, m, n, r, alpha, initial_seed; doPYnnm=true, doSCSnnm=false)
            resstats = get_stats(pyres)
            curmethod = "PYnnm"
        else
            scsres = nnm_test(k, m, n, r, alpha, initial_seed; doPYnnm=false, doSCSnnm=true)
            resstats = get_stats(scsres)
            curmethod = "SCSnnm"
        end
        return DataFrame(
            k=k, m=m, n=n, r=r, alpha=alpha, initial_seed=initial_seed,
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

# RUN TESTS (SQUARE n x n)

n_range = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
alpha_range = [i for i in 0.05:0.05:0.95]
# h(n) = n < 100 ? 1 : n / 100
# derive_r_range(n) = sort(unique([Int(round(n/i)) for i in 2:h(n):n])) # local
derive_r_range(n) = [i for i in 1:min(div(n, 2), 30)] # on HPC

df = DataFrame(
    k=Int[], m=Int[], n=Int[], r=Int[], alpha=Float64[], initial_seed=Int[],
    method=String[], metric=String[], mean=Float64[], std=Float64[], 
    min=Float64[], median=Float64[], max=Float64[], pct_success=Float64[])

for n in n_range[2]
    k = n < 100 ? 100 : 20  
    for alpha in alpha_range
        for r in derive_r_range(n)
            data = record_stats(k, n, n, r, alpha, 892143)
            append!(df, data)
            println("Finished r = ", r, " at ", Dates.format(now(), "HH:MM:SS"))
        end
        println("Finished alpha = ", alpha, " at ", Dates.format(now(), "HH:MM:SS"))
    end
    println("Finished n = ", n, " at ", Dates.format(now(), "HH:MM:SS"))
end

# Write df to csv
# make sure directory exists; if not, create it
subfolder = "data"
if !isdir(subfolder)
    mkdir(subfolder)
end
CSV.write("$(subfolder)/baseline_nnm_results_2.csv", df, append=false)

# Read df from csv
#df = CSV.read("data/baseline_nnm_results_2.csv", DataFrame)


# For a given matrix size, create three heatmap plots, each with r on the x-axis and alpha on the y-axis. 
# The three plots should show the mean time, mean RMSE, and the percentage of success.
f = Figure(size=(1200, 400))
titles = ["Mean time (in log seconds)", "O(mean RMSE) for unobserved entries", "Proportion of success (tol = 1e-4)"]
selectedcols = [:mean, :mean, :pct_success]
metricfilters = ["time", "RMSE", "RMSE"]
for i in 1:3
    plotdata = filter(row -> row.metric == metricfilters[i], df)
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
                 tickformat = values -> ["1e$(Int(value))" for value in values])
    else
        ax, hm = contourf(
            f[1, i][1,1], plotdata[!, :r], plotdata[!, :alpha], zvalues, colormap = :viridis,
            visible=true)
        Colorbar(f[:, i][1,2], hm, vertical = true)
    end
    ax.title = titles[i]
    ax.xlabel = "Rank"
    ax.xticks = 1:5:maximum(plotdata[!, :r])
    ax.ylabel = "Proportion of known entries"
end
Label(f[0, :], text = "Performance of low-rank matrix completion via nuclear norm minimization\n"*
                      "over $(df[1, :k]) random n x n matrices of size n = $(df[1, :n])", 
      fontsize = 24)
f
save("$(subfolder)/baseline_nnm_results_$(df[1, :n]).png", f)
