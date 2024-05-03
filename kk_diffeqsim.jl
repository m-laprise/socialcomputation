using LinearAlgebra, Random
using DifferentialEquations
using Statistics
using DiffEqBase
using Glob
using PyPlot, PyCall, BenchmarkTools, Printf

# PyPlot.rc("text", usetex=true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
#rcParams["text.latex.preamble"] = ["\\usepackage[utf8]{inputenc}", "\\usepackage{amsmath,amsthm}"];
rcParams["pdf.fonttype"] = [42];
rcParams["ps.fonttype"] = [42];
rcParams["savefig.transparent"] = true;
rcParams["text.usetex"] = true;
PyPlot.rc("font", size = 12)          # set default fontsize
PyPlot.rc("figure", figsize = (6, 4)) # set default figsize
PyPlot.svg(true);


#Helper functions
function phi_h(g,b,x)
    return tanh(g*x + b) 
 end
 
 function d_phi_h(g,b,x)
    return  g /(cosh(g*x + b))^2
 end
 
 function d_sig_z(a,b,z)
     return sig_z(a,b,z)*(1 - sig_z(a,b,z))*a
 end

# Gate with infinite az
function sig_z(a,b,z)
    return 0.5*(1 + sign(z+b))
end   

struct opinionRNN_param_struct{T}
    d::T; gu::T; tau_u ::T; 
    Jh::AbstractMatrix{T}
    Ju::AbstractMatrix{T}
    u_init::AbstractVector{T}
    aux_Jh::AbstractVector{T}
    aux_UJh::AbstractVector{T}
    aux_JuH::AbstractVector{T}
    h_input::AbstractVector{T}
end

# u = (h)
# in-place equation of motion
function opinionRNN_eom!(du, u, p::opinionRNN_param_struct, t)
    gu = p.gu; d = p.d; tau_u = p.tau_u; Jh = p.Jh; Ju = p.Ju;  h_input = p.h_input;
    aux_Jh = p.aux_Jh;  gu = p.gu; d = p.d; aux_UJh = p.aux_UJh; aux_JuH = p.aux_JuH;
    u_init = p.u_init
    n::Int = Int(size(u)[1]/2)
    _h = @view u[1:n]; _dh = @view du[1:n]; _u = @view u[n+1:end]; _du = @view du[n+1:end]
    mul!(aux_Jh,Jh,_h)
    @. aux_UJh = sig_z(1,0,_u*aux_Jh)
    mul!(aux_JuH,Ju,_h.^2)
    @. _dh = -d*_h + aux_UJh + h_input
    @. _du = (1/tau_u)*(-_u + u_init + gu*aux_JuH) 
    return nothing
end


rnd_seed = 12344
Random.seed!(rnd_seed);
n = 500;  # N = 2*n;
#These are needed for all models
d =0.5; gu = 1.0; tau_u = 1.0;
t_total = 50.0
dt = 1.0
t0 = 0.0;
t1 = t_total;
t_offset1 = Int(round(t0/dt))
t_offset2 = Int(round(t1/dt))
rnd_idx = 1:n
t_trace = t_offset1:1:t_offset2;


# Specify the parameters:
Jh = randn(n,n)/sqrt(n);  
Ju = randn(n,n)/sqrt(n);

# initial state vectors
du0 = randn(2*n,);
u0 = randn(2*n,);

params = opinionRNN_param_struct(d,gu,tau_u,Jh,Ju,zeros(n,),zeros(n,),zeros(n,),zeros(n,),zeros(n,));
#(gh,bh,az,bz,G1,G2,Jh,Jz,rand(n,),rand(n,),rand(n,),rand(n,),rand(n,),zeros(n,));

# This is where you choose which model you want : vXX?
opinionRNN_fun = ODEFunction(opinionRNN_eom!)
opinionRNN_prob = ODEProblem(opinionRNN_fun,u0,(0.0,t_total),params);

# Integrate away!
rnd_idx = rand(1:n,10)
opinionRNN_soln = solve(opinionRNN_prob,Tsit5(); abstol = 1e-8, reltol = 1e-3,
                        saveat=dt);  # Vern9()  takes almost 2x the time
h = opinionRNN_soln[1:n,:]';
u = opinionRNN_soln[n+1:end,:]';

size(opinionRNN_soln)


fig2, f2_axes = plt.subplots(ncols=1, nrows=1,figsize=(8,6))


ax1 = f2_axes
rnd_idx = rand(1:n,10)
h_plot = h[:,rnd_idx]
ax1.plot(t_trace,h_plot,linewidth=0.5)
#ax1.set_ylabel(L"$h$",fontsize=16)
y_min = -1.5
y_max = 1.5
ax1.set_ylim((y_min,y_max))
ax1.set_yticks([-1,0,1])
ax1.set_xlim((t0,t1))
ax1.tick_params(labelsize=18)

#fig2.tight_layout()
show()

plt.plot(t_trace,h_plot,linewidth=0.5)
PyPlot.display_figs()