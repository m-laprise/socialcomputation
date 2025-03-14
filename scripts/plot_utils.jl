using LinearAlgebra
using CairoMakie

function ploteigvals(A::AbstractMatrix)
    eigs = eigvals(A)
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Eigenvalues", xlabel = "Real", ylabel = "Imaginary")
    θ = LinRange(0, 2π, 1000)
    circle_x = cos.(θ)
    circle_y = sin.(θ)
    lines!(ax, circle_x, circle_y, color = :black)
    scatter!(ax, real(eigs), imag(eigs), color = :blue)
    return fig
end

function ploteigvals(eigs::Vector)
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Eigenvalues", xlabel = "Real", ylabel = "Imaginary")
    θ = LinRange(0, 2π, 1000)
    circle_x = cos.(θ)
    circle_y = sin.(θ)
    lines!(ax, circle_x, circle_y, color = :black)
    scatter!(ax, real(eigs), imag(eigs), color = :blue)
    return fig
end

function ploteigvals!(ax, A::AbstractMatrix; kwargs...)
    eigs = eigvals(A)
    θ = LinRange(0, 2π, 1000)
    circle_x = cos.(θ)
    circle_y = sin.(θ)
    lines!(ax, circle_x, circle_y, color = :black)
    scatter!(ax, real(eigs), imag(eigs), color = :blue, kwargs...)
end

function ploteigvals!(ax, eigs::Vector; kwargs...)
    θ = LinRange(0, 2π, 1000)
    circle_x = cos.(θ)
    circle_y = sin.(θ)
    lines!(ax, circle_x, circle_y, color = :black)
    scatter!(ax, real(eigs), imag(eigs), color = :blue, kwargs...)
end

function main_training_figure(train_metrics, val_metrics, test_metrics, 
                              dataX, tasklab, modlabel, 
                              hidden_dim, dec_rank, init_eta, 
                              eta_period, minibatch_size, k, turns)
    fig = Figure(size = (850, 1200))
    epochs = length(val_metrics[:loss])-1
    #train_metrics[:spectralgapovernorm] = round.(train_metrics[:spectralgap] ./ train_metrics[:spectralnorm], digits=2)
    #train_metrics[:logloss] = log.(train_metrics[:loss])
    #val_metrics[:logloss] = log.(val_metrics[:loss])
    #test_metrics[:logloss] = log.(test_metrics[:loss])
    metrics = [
        (1, 1:2, "Loss", "loss", "Mean spectrum-penalized Huber loss (known entries)"),
        #(1, 2, "RMSE", "all_rmse", "RMSE (all entries)"),
        (2, 1, "MAE", "known_mae", "Mean MAE (known entries)"),
        (2, 2, "MAE", "all_mae", "Mean MAE (all entries)"),
        #(3, 1, "Spectral gap / spectral norm", "spectralgapovernorm", "Mean spectral gap / mean spectral norm"),
        #(3, 2, "Spectral gap", "spectralgap", "Mean spectral gap"),
        (3, 1, "Nuclear norm", "nuclearnorm", "Mean nuclear norm"),
        (3, 2, "Variance", "variance", "Mean variance of matrix entries")
    ]
    for (row, col, ylabel, key, title) in metrics
        ax = Axis(fig[row, col], xlabel = "Epochs", ylabel = ylabel, title = title)
        if key in ["loss"]
            lines!(ax, [i for i in range(0, epochs, length(train_metrics[Symbol(key)]))], #[length(dataloader)+1:end]))], 
                train_metrics[Symbol(key)], #[length(dataloader)+1:end], 
                color = :blue, label = "Training")
            scatter!(ax, 0:epochs, val_metrics[Symbol(key)],#[2:end], 
                color = :red, markersize = 6, label = "Validation")
            lines!(ax, 0:epochs, [test_metrics[Symbol(key)][end]], color = :green, linestyle = :dash)
            scatter!(ax, epochs, test_metrics[Symbol(key)][end], color = :green, label = "Final Test")
        elseif key in ["known_mae", "all_mae", "all_rmse"]
            lines!(ax, 0:epochs, train_metrics[Symbol(key)],#[2:end], 
            color = :blue, label = "Training")
            lines!(ax, 0:epochs, val_metrics[Symbol(key)],#[2:end], 
            color = :red, label = "Validation")
            lines!(ax, 0:epochs, [test_metrics[Symbol(key)][end]], color = :green, linestyle = :dash)
            scatter!(ax, epochs, test_metrics[Symbol(key)][end], color = :green, label = "Final Test")
        else
            lines!(ax, 0:epochs, train_metrics[Symbol(key)],#[2:end], 
            color = :blue, label = "Reconstructed")
            lines!(ax, 0:epochs, [eval(Symbol("ref$key"))], color = :orange, linestyle = :dash, label = "Mean ground truth")
        end
        axislegend(ax, backgroundcolor = :transparent)
    end
    Label(
        fig[begin-1, :],
        "$(tasklab)\n$(modlabel) RNN of $(k) units, $(turns) dynamic steps"*
        "\nwith training loss based on known entries",
        fontsize = 20,
        padding = (0, 0, 0, 0),
    )
    Label(
        fig[end+1, :],
        "Optimizer: Adam with schedule CosAnneal(start = $(init_eta), period = $(eta_period))\n"*
        "for $(epochs) epochs over $(size(dataX, 3)) examples, minibatch size $(minibatch_size).\n"*
        "Hidden internal state dimension: $(hidden_dim). Enforced decoder rank: $(dec_rank).\n"*
        "Test loss (known entries): $(round(test_metrics[:loss][end], digits=4)).\n"*
        "Test MAE (known entries): $(round(test_metrics[:known_mae][end], digits=4)).\n"*
        "Test MAE (all entries): $(round(test_metrics[:all_mae][end], digits=4)).\n",
        fontsize = 14,
        padding = (0, 0, 0, 0),
    )
    fig
end
