using RDatasets
using DataFrames
using LinearAlgebra
using CategoricalArrays
using Distributions
using CairoMakie
CairoMakie.activate!(type = "png")

alldf = RDatasets.datasets()
# Filter out when column "Rows" is < 500
alldf = alldf[alldf[!, :Rows] .> 500, :] 
alldf = alldf[alldf[!, :Columns] .> 5, :]
alldf[!, :TotalObs] = alldf[!, :Rows] .* alldf[!, :Columns]
# Sort by "TotalObs" in descending order
alldf = sort!(alldf, :TotalObs, rev=true)

# Get information about each unique ":Package" values
pkgs = RDatasets.packages()
pkgs = pkgs[pkgs[!, :Package] .∈ Ref(unique(alldf[!, :Package])), :]

iris = RDatasets.dataset("datasets", "iris")
mood = RDatasets.dataset("psych", "msq")
movies = RDatasets.dataset("ggplot2", "movies")
# remove :Budget and :Title columns from movies dataset
movies = select!(movies, Not([:Budget, :Title, :MPAA]))
diamonds = RDatasets.dataset("ggplot2", "diamonds")
vietnam = RDatasets.dataset("Ecdat", "VietNamI")
vietnam[!, :Female] = ifelse.(vietnam[!, :Sex] .== "Female", 1, 0)
vietnam = select!(vietnam, Not([:Sex]))

carchoice = RDatasets.dataset("Ecdat", "Car")
baseball = RDatasets.dataset("plyr", "baseball")
tic = RDatasets.dataset("lme4", "Caravan")

# Function: Inspect dataset. If categorical column(s) exist, transform into dummy variables (one-hot encoding).
function cat_to_dummies(df::DataFrame)
    for col in names(df)
        if isa(df[!, col], CategoricalArray)
            # Create dummy for each category
            dummies = DataFrame()
            for cat in levels(df[!, col])
                dummies[!, Symbol(col, "_", cat)] = ifelse.(df[!, col] .== cat, 1, 0)
            end
            # Append dummies to original DataFrame
            df = hcat(df, dummies)
            # Remove original column
            df = select!(df, Not(col))
        end
    end
    return df
end

function handle_missing(df::DataFrame, threshold::Float64)
    nrows = nrow(df)
    df = df[:, [col for col in names(df) if count(ismissing, df[!, col]) / nrows <= threshold]]
    return dropmissing(df)
end

function get_and_show_singularvalues(df::DataFrame, logscale::Bool = true)
    df = handle_missing(df, 0.1)
    df = cat_to_dummies(df)
    X = Matrix(df)
    X = X .- mean(X, dims=1)
    u, s, vt = svd(X)
    sum_s = sum(s[2:end])
    ratio = s[1] / sum_s
    fig = Figure(size = (800, 800))
    fig[1, 1] = Label(fig, text = "Singular values of dataset (log)", 
                      fontsize = 24, tellwidth = false, halign = :center, padding = (0, 0, 10, 0))
    ax = Axis(fig[2, 1], title = "Singular values")
    s = sort(s, rev=true)
    if logscale
        s = log10.(s)
    end
    barplot!(ax, s, show_axis = false, show_grid = false)
    fig
    return fig, ratio
end
fig, ratio = get_and_show_singularvalues(diamonds)
fig, ratio = get_and_show_singularvalues(mood)
fig, ratio = get_and_show_singularvalues(movies)
fig, ratio = get_and_show_singularvalues(iris)

# Function: Take dataset as input. Convert categorical columns into dummy variables. Convert result into a matrix.
# Manually derive variance-covariance matrix and correlation matrix 
function get_cov_corr_matrix(df::DataFrame)
    df = handle_missing(df, 0.1)
    df = cat_to_dummies(df)
    X = Matrix(df)
    # Center all variables
    X = X .- mean(X, dims=1)
    n = size(X, 1)
    cov_matrix = (X'*X) / n
    corr_matrix = cov_matrix ./ sqrt.(diag(cov_matrix) * diag(cov_matrix)')
    return cov_matrix, corr_matrix
end


iris_cov, iris_corr = get_cov_corr_matrix(iris)
mood_cov, mood_corr = get_cov_corr_matrix(mood)
movies_cov, movies_corr = get_cov_corr_matrix(movies)
diamonds_cov, diamonds_corr = get_cov_corr_matrix(diamonds)

# Display visualization of correlation matrices for each dataset, 2x2 grid, using heatmap
using CairoMakie
CairoMakie.activate!(type = "png")

fig = Figure(size = (800, 800))
fig[1, 1] = Label(fig, text = "Correlation Matrices", 
                  fontsize = 24, tellwidth = false, halign = :center, padding = (0, 0, 10, 0))
grid = fig[2, 1] = GridLayout()
titles = ["Iris", "Mood", "Movies", "Diamonds"]
corrs = [iris_corr, mood_corr, movies_corr, diamonds_corr]
for (i, title, corr) in zip(1:4, titles, corrs)
    ax = Axis(grid[i ÷ 3 + 1, i % 2 + 1], title = title)
    heatmap!(ax, corr, colormap = :viridis, show_axis = false, show_grid = false, colorrange = (0, 1))
end
fig
save("correlation_matrices.png", fig)

# Plot histogram of covariance matrix values for each dataset, omitting diagonal values
fig = Figure(size = (800, 800))
fig[1, 1] = Label(fig, text = "Distribution of Correlation Values\n(diagonal omitted)", 
                  fontsize = 24, tellwidth = false, halign = :center, padding = (0, 0, 10, 0))
grid = fig[2, 1] = GridLayout()
titles = ["Iris", "Mood", "Movies", "Diamonds"]
corrs = [iris_corr, mood_corr, movies_corr, diamonds_corr]
for (i, title, corr) in zip(1:4, titles, corrs)
    ax = Axis(grid[i ÷ 3 + 1, i % 2 + 1], title = title)
    trils = tril(corr, -1)
    corr_values = trils[trils .!= 0]
    hist!(ax, corr_values, bins = 50, show_axis = false, show_grid = false)
    xlims!(-1, 1)
end
fig
save("correlation_histograms.png", fig)


function get_eigenvalues(cov_matrix::Matrix)
    eigvals = eigen(cov_matrix).values
    return eigvals
end



# Display visualization of eigenvalues for each covariance, 2x2 grid, using bar chart
iris_eigvals = get_eigenvalues(iris_cov)
mood_eigvals = get_eigenvalues(mood_cov)
movies_eigvals = get_eigenvalues(movies_cov)
diamonds_eigvals = get_eigenvalues(diamonds_cov)

fig = Figure(size = (800, 800))
fig[1, 1] = Label(fig, text = "Eigenvalues of covariance matrix (log scale)", 
                  fontsize = 24, tellwidth = false, halign = :center, padding = (0, 0, 10, 0))
grid = fig[2, 1] = GridLayout()
titles = ["Iris", "Mood", "Movies", "Diamonds"]
eigvals = [iris_eigvals, mood_eigvals, movies_eigvals, diamonds_eigvals]
for (i, title, eigval) in zip(1:4, titles, eigvals)
    ax = Axis(grid[i ÷ 3 + 1, i % 2 + 1], title = title)
    eigval = eigval[eigval .> 1e-10]
    eigval = sort(eigval, rev=true)
    eigval = log10.(eigval)
    barplot!(ax, eigval, show_axis = false, show_grid = false)
end
fig
save("cov_eigenvalues.png", fig)

# Same for correlation matrices
iris_eigvals = get_eigenvalues(iris_corr)
mood_eigvals = get_eigenvalues(mood_corr)
movies_eigvals = get_eigenvalues(movies_corr)
diamonds_eigvals = get_eigenvalues(diamonds_corr)

fig = Figure(size = (800, 800))
fig[1, 1] = Label(fig, text = "Eigenvalues of correlation matrix (log scale)", 
                  fontsize = 24, tellwidth = false, halign = :center, padding = (0, 0, 10, 0))
grid = fig[2, 1] = GridLayout()
titles = ["Iris", "Mood", "Movies", "Diamonds"]
eigvals = [iris_eigvals, mood_eigvals, movies_eigvals, diamonds_eigvals]
for (i, title, eigval) in zip(1:4, titles, eigvals)
    ax = Axis(grid[i ÷ 3 + 1, i % 2 + 1], title = title)
    eigval = eigval[eigval .> 1e-10]
    eigval = sort(eigval, rev=true)
    eigval = log10.(eigval)
    barplot!(ax, eigval, show_axis = false, show_grid = false)
end
fig
save("corr_eigenvalues.png", fig)