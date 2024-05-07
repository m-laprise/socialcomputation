using DifferentialEquations
using Graphs
using RecursiveArrayTools

# INITIALIZE ABM AND RETRIEVE PARAMETERS AND INITIAL CONDITIONS
include("BFL_discretetime_mod.jl")
numagents = 100
budparam = 4
seed = 3782
model, g = init_bfl_model(numagents=numagents,
                          jz_network = ("erdosrenyi", budparam),
                          gating=true, 
                          basal_scp=0.0,
                          seed=seed, 
                          #alpha_gate=2.0, beta_gate=0.0
                          dampingparam=(0.75, 0.25), 
                          scalingparam=(1, 0.25))
# Parameters
## Connectivity matrix
adj_g = adjacency_matrix(g) + 1I(numagents)
adj_g = Matrix(adj_g)
## Unit specific parameters
d = [i.damping for i in allagents(model)]
k = [i.scaling for i in allagents(model)]
## Model wide parameters
u0 = [model.basal_scp for i in allagents(model)]
gu = [model.gain_scp for i in allagents(model)]
tauu = [model.tau_scp for i in allagents(model)]
taux = [model.tau_gate for i in allagents(model)]
alpha = [model.alpha_gate for i in allagents(model)]
beta = [model.beta_gate for i in allagents(model)]
## Parameter vector
ng_p = ArrayPartition(d, u0, gu, tauu, adj_g)
p = ArrayPartition(d, k, u0, gu, tauu, taux, alpha, beta, adj_g)

# Initial conditions
z = [i.opinion_new for i in allagents(model)]
u = [i.scp_state for i in allagents(model)]
x = [i.gating_state for i in allagents(model)]
ng_v0 = ArrayPartition(z, u)
v0 = ArrayPartition(z, u, x)

function gatedsoccomputation!(dv, v, p, t)
    z, u, x = v.x[1], v.x[2], v.x[3]
    d, k, u0, gu, tauu, taux, a, b, adj_g = p.x[1], p.x[2], p.x[3], p.x[4], p.x[5], p.x[6], p.x[7], p.x[8], p.x[9]
    inputs = zeros(length(z))
    dv.x[1] .= (ones(length(z)) ./ (ones(length(z)) .+ exp.(.-a.*x .+ b)) ) .* (.-d .* z .+ tanh.(adj_g*z)) .+ inputs
    dv.x[2] .= (.-u .+ u0 .+ gu .* (adj_g* z.^2)) ./ tauu
    dv.x[3] .= (.-x .+ adj_g*tanh.(z) .+ k.*inputs) ./ taux
end

tspan = (0.0, 20.0)
prob_wg = ODEProblem(gatedsoccomputation!, v0, tspan, p)
sol_wg = solve(prob_wg, AutoTsit5(Rosenbrock23()))
plot(sol_wg)

function soccomputation!(dv, v, p, t)
    z, u = v.x[1], v.x[2]
    d, u0, gu, tauu, adj_g = p.x[1], p.x[2], p.x[3], p.x[4], p.x[5]
    inputs = zeros(length(z))
    dv.x[1] .= (.-d .* z .+ tanh.(adj_g*z)) .+ inputs
    dv.x[2] .= (.-u .+ u0 .+ gu .* (adj_g* z.^2)) ./ tauu
end

prob_nog = ODEProblem(soccomputation!, ng_v0, tspan, ng_p)
sol_nog = solve(prob_nog, AutoTsit5(Rosenbrock23()))
plot(sol_nog)

plot(LinRange(minimum(sol_nog.t), maximum(minimum(sol_nog.t)), 1000))
plot(sol_nog.t, sol_nog[:,1,:][2,:])

using Plots

xyzt = Plots.plot(sol_nog, lw = 1.5, legend = false)
xy = Plots.plot(sol_nog, plotdensity = 10000, idxs = (1, 2), legend = false)
xz = Plots.plot(sol_nog, plotdensity = 10000, idxs = (1, 3), legend = false)
yz = Plots.plot(sol_nog, plotdensity = 10000, idxs = (2, 3), legend = false)
xyz = Plots.plot(sol_nog, plotdensity = 10000, idxs = (1, 2, 3), legend = false)
Plots.plot(Plots.plot(xyzt, xyz), Plots.plot(xy, xz, yz, layout = (1, 3), w = 1), layout = (2, 1), legend = false)



using GraphMakie
using NetworkLayout
f, ax, p = graphplot(g; #layout=Spring(dim=2, seed=5),
                        node_marker=Circle, 
                        #nlabels=repr.(vertices(g)),
                        #nlabels_color=:black, nlabels_fontsize=11,
                        node_size=[sqrt(degree(g, i))*3 for i in vertices(g)], 
                        node_color=[:black for i in vertices(g)], 
                        edge_color=:grey32, line_width=0.5)
hidedecorations!(ax); hidespines!(ax)
f