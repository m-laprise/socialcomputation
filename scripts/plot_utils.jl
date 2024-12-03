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

