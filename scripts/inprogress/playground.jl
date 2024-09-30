
include("genrandommatrix.jl")


n = 50
d = 1000
# Identity baseline
baselineX = gen_data_from_vcov(Matrix{Float64}(I, n, n), d)
baselineΣhat1 = cov(baselineX, corrected=true)
sum(abs.(baselineΣhat1 .- Matrix{Float64}(I, n, n))) / (n*n/2)

# VERY LOW RANK
r = 2
llrΣ = gen_symm_matrix(n, r, sparse = true)
llrX = gen_data_from_vcov(Matrix(llrΣ), d)
llrΣhat1 = cov(llrX, corrected=true)
sum(abs.(llrΣ .- llrΣhat1)) / (n*n/2)

# LOW RANK
r = 10
lrΣ = gen_symm_matrix(n, r, sparse = true)
lrX = gen_data_from_vcov(Matrix(lrΣ), d)
lrΣhat1 = cov(lrX, corrected=true)
sum(abs.(lrΣ .- lrΣhat1)) / (n*n/2)

# FULL RANK
frΣ = gen_symm_matrix(n, n, sparse = true)
frX = gen_data_from_vcov(Matrix(frΣ), d)
frΣhat1 = cov(frX, corrected=true)
sum(abs.(frΣ .- frΣhat1)) / (n*n/2)


using CairoMakie

# ORGANIZE DATA
cov_trueeigvals = [eigvals(llrΣ), eigvals(lrΣ), eigvals(frΣ), eigvals(Matrix{Float64}(I, n, n))]
cov_trueeigvals = [eigvals[eigvals .> 1e-10] for eigvals in cov_trueeigvals]
cov_estimeigvals = [eigvals(llrΣhat1), eigvals(lrΣhat1), eigvals(frΣhat1), eigvals(baselineΣhat1), ]
cov_estimeigvals = [eigvals[eigvals .> 1e-10] for eigvals in cov_estimeigvals]
# paramaters
myalpha = 0.75
mymarkersize = 6
# titles and labels
legendlabs = ["True Σ Spectrum", "Σ-hat Spectrum"]
horizontal = ["Rank 2", "Rank 10", "Rank 50 (full rank)", "Identity (independent)"]
vertical = ["Covariance"]
# BETTER LOOKING FIGURE, with 4 x 1 panels
fig = Figure(resolution = (650, 900))
fig[1:4, 1] = [Axis(fig, title = horizontal[i], yscale = log10) for i in 1:4]
supertitle = Label(
    fig[0, :], "Spectrum of true and estimated\nd x d covariance matrices (n = 1000 samples)", 
    fontsize = 24)
sideinfo = Label(fig[1:4, 0], "Eigenvalues (log scale)", rotation = pi/2)
for i in 1:4
    ax = fig[i, 1]
    line1 = plot!(ax, cov_trueeigvals[i], alpha = myalpha, markersize = mymarkersize, color = :blue)
    line2 = plot!(ax, cov_estimeigvals[i], alpha = myalpha, markersize = mymarkersize, color = :red)
end
legend = Legend(fig, [line1, line2], legendlabs, position = (0.1, 0.9))
fig

# Sum absolute values over rows of sigma, find index of row with the largest sum
function find_max_row(A)
    sums = sum(abs.(A), dims=2)
    maxidx = argmax(sums)
    return maxidx[1]
end

y_llr = llrX[:,find_max_row(llrΣ)]
X_llr = copy(llrX[:,1:end .!= find_max_row(llrΣ)])

y_lr = lrX[:,find_max_row(lrΣ)]
X_lr = copy(lrX[:,1:end .!= find_max_row(lrΣ)])

y_fr = frX[:,find_max_row(frΣ)]
X_fr = copy(frX[:,1:end .!= find_max_row(frΣ)])

randidx = rand(1:n)
y_baseline = baselineX[:,randidx]
X_baseline = copy(baselineX[:,1:end .!= randidx])


# GLM

rsquared_llr, testerror_llr, model_llr = fit_linear_model(X_llr, y_llr)
null_model_llr = lm(@formula(y ~ 1), DataFrame(y=y_llr))
rsquared_lr, testerror_lr, model_lr = fit_linear_model(X_lr, y_lr)
null_model_lr = lm(@formula(y ~ 1), DataFrame(y=y_lr))
rsquared_fr, testerror_fr, model_fr = fit_linear_model(X_fr, y_fr)
null_model_fr = lm(@formula(y ~ 1), DataFrame(y=y_fr))
rsquared_baseline, testerror_baseline, model_baseline = fit_linear_model(X_baseline, y_baseline)
null_model_baseline = lm(@formula(y ~ 1), DataFrame(y=y_baseline))

# Display results
println("BASELINE (Identity covariance):")
println("Full linear model: R^2 = ", rsquared_baseline, ", test error = ", testerror_baseline)
println("Null model: test error = ", sum((y_baseline .- mean(y_baseline)).^2) / length(y_baseline))

println("VERY LOW RANK:")
println("Full linear model: R^2 = ", rsquared_llr, ", test error = ", testerror_llr)
println("Null model: test error = ", sum((y_llr .- mean(y_llr)).^2) / length(y_llr))

println("LOW RANK:")
println("Full linear model: R^2 = ", rsquared_lr, ", test error = ", testerror_lr)
println("Null model: test error = ", sum((y_lr .- mean(y_lr)).^2) / length(y_lr))

println("FULL RANK:")
println("Full linear model: R^2 = ", rsquared_fr, ", test error = ", testerror_fr)
println("Null model: test error = ", sum((y_fr .- mean(y_fr)).^2) / length(y_fr))