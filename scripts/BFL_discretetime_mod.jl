# Julia implementation of the BFL model for multi-agent opinion formation (Leonard et al. 2023)
# Discrete time version
# Code author: Marie-Lou Laprise
# This code was written for Julia 1.10.2 and Agents 6.0.7
using Agents
using CairoMakie
using DataFrames
using Graphs
using Random
using LinearAlgebra
using SparseArrays
using Statistics

# Create agent type.
# The agent type is defined as a struct with fields for the properties of the agent.
# (Note that we could make the three opinions a single field with vector value.)
# In future implementations, `damping` and `personalbias` can be set to change over time or be agent-specific.
"""
    struct OpinionatedGuy <: NoSpaceAgent

The `OpinionatedGuy` struct represents an agent with opinionated behavior. It is a subtype of the `NoSpaceAgent` struct.

# Fields
- `opinion_temp::Float64`: The temporary opinion of the agent, used for synchronous agent update (value needed at the start and end of the step).
- `opinion_new::Float64`: The new opinion of the agent, calculated at the end of the step. 
- `opinion_prev::Float64`: The previous opinion of the agent, required to check convergence for termination.
- `scp_state::Float64`: The susceptibility state of the agent.
- `damping::Float64`: The damping factor of the agent for the opinion update. By default, it will be sampled from a Gaussian.
- `scaling::Float64`: The scaling factor for the agent. By default, it will be sampled from a Gaussian.
- `personalbias::Float64`: The bias term for the agent. By default, it will be inherited from the model and set to 0.
"""
@agent struct OpinionatedGuy(NoSpaceAgent)
    opinion_temp::Float64
    opinion_new::Float64
    opinion_prev::Float64
    scp_state::Float64
    gating_state::Float64
    damping::Float64
    scaling::Float64
    personalbias::Float64
end

"""
    init_bfl_model(; numagents, damping, basal_scp, gain_scp,
                    tau_scp, saturation, biastype, 
                    jz_network, ju_network, opinioninit, maxsteps, seed) -> model::ABM

Create an ABM model for the BFL simulation. It is a keyword function that initializes 
the model with the specified parameters, which can be used for parameter scanning.

# Arguments
- `numagents::Int`: The number of agents in the model.
- `damping::Float64`: The damping factor for the opinion update. 
- `basal_scp::Float64`: The basal susceptibility state of the agents. 
- `tau_scp::Float64`: The time constant for the susceptibility state update. 
- `saturation::String`: The saturation function for the susceptibility state. Default is "tanh" (hyperbolic tangent for opinions between -1 and 1).
- `biastype::String`: The bias function. Default is "randomsmall" (small random bias for each agent at each step).
- `biasproc::Vector{Float64}`: The evolution vector for the bias calculation if biastype = "manual". Default is a vector of random numbers.
- `numsensing::Int`: The number of agents that are sensitive to external information. Default is 0.
- `jz_network::Tuple`: The parameters for the communication network. Default is ("barabasialbert", 2) (Barabasi-Albert network with m=2).
- `ju_network::String`: The susceptibility network type. 
- `opinioninit::String`: The opinion initialization method. 
- `maxsteps::Int`: The maximum number of steps to run the model. Default is 500. 0 means run until convergence.
- `euler_h::Float64`: The Euler integration step size for the opinion and susceptibility state updates. Default is 0.1.
- `seed::Int`: The seed for the random number generator. Default is 23.

# Returns
- `model::ABM`: The created ABM model.
"""
function init_bfl_model(; numagents::Int = 100,
                        maxsteps::Int = 500,
                        basal_scp::Float64 = 0.01, 
                        gain_scp::Float64 = 1.0,
                        tau_scp::Float64 = 1.0,
                        saturation::String = "tanh",
                        jz_network::Tuple = ("barabasialbert", 3),
                        ju_network::String = "buddies",
                        # Unit-specific parameters
                        dampingtype::String = "random", 
                        dampingparam::Tuple = (0.5, 0.25),
                        scalingtype::String = "random",
                        scalingparam::Tuple = (1.0,0.5),
                        # External input
                        inputmatrix::AbstractMatrix{Float64} = zeros(numagents, maxsteps),
                        noisematrix::AbstractMatrix{Float64} = zeros(numagents, maxsteps),
                        #biastype::String = "randomsmall",
                        #biasproc::Vector{Float64} = zeros(maxsteps),
                        #numsensing::Int = 0,
                        #grsensing::Int = 0,
                        # Gate parameters
                        gating::Bool = false,
                        jx_network::String = "2degree",
                        tau_gate::Float64 = 1.0,
                        alpha_gate::Float64 = 1.0,
                        beta_gate::Float64 = 0.0,
                        # Initialization and simulation
                        opinioninit::String = "random",
                        euler_h::Float64 = 0.1,
                        seed::Int=23)
    # Check that input matrix is numagents x maxsteps, otherwise error
    #if size(biasproc) != (maxsteps,)
    #    error("Bias process vector must be of length maxsteps.")
    #end
    if size(inputmatrix) != (numagents, maxsteps) || size(noisematrix) != (numagents, maxsteps)
        error("Input and noise matrices must be of size numagents x maxsteps.")
    end
    # Create an ABM model with the OpinionatedGuy agent type
    buddies, adj_ju, adj_jx = network_generation(jz_network, ju_network, jx_network, numagents, seed)
    # Print some basic statistics about the resulting social graph buddies
    println("Social graph initialized with $(jz_network[1]) network.")
    println("Number of edges: $(ne(buddies))")
    avgdeg = mean(Graphs.degree(buddies))
    println("Average degree: $(avgdeg)")
    println("Maximum degree: $(maximum(Graphs.degree(buddies)))")
    println("Edge density: $(Graphs.density(buddies))")
    println("Assortativity coef: $(assortativity(buddies))")
    println("Global clustering coef: $(global_clustering_coefficient(buddies))")
    println("Rich-club coef, k=mean: $(rich_club(buddies, Int(round(avgdeg))))")
    println("Vertices with self-loops? $(has_self_loops(buddies))")
    println("Graph is connected? $(is_connected(buddies))")
    # core_periphery_deg(test)
    model = StandardABM(OpinionatedGuy; agent_step!, model_step!, 
        rng = MersenneTwister(seed), scheduler = Schedulers.fastest,
        properties = Dict(
            :basal_scp => basal_scp,
            :gain_scp => gain_scp,
            :tau_scp => tau_scp,
            :saturation => saturation,
            #:biastype => biastype,
            #:biasproc => biasproc,
            :buddies => buddies, # communication graph
            :adj_ju => adj_ju, # susceptibility graph
            # External input
            :inputmatrix => inputmatrix,
            :noisematrix => noisematrix,
            # Gating parameters
            :gating => gating,
            :adj_jx => adj_jx, # gating graph
            :tau_gate => tau_gate,
            :alpha_gate => alpha_gate,
            :beta_gate => beta_gate,
            # Simulation
            :maxsteps => maxsteps,
            :euler_h => euler_h,
        )
    )
    # Select agents to be sensitive to external information
    #if numsensing > 0
        #sensors = randperm(numagents)[1:numsensing]
        #sensors_odd = sensors[1:2:end]
        #sensors_even = sensors[2:2:end]
    #end
    # Add numagents agents to the model
    for i in 1:numagents
        personalbias = 0.0
        # Set opinion_temp, opinion_new, opinion_prev to same random initial value
        o = init_opinion(model, i, opinioninit)
        # Set initial gating state to zero
        x = 0.0
        # Set initial susceptibility to basal, set damping and scaling
        d, k = init_unitparam(model, i, dampingtype, dampingparam, scalingtype, scalingparam)
        add_agent!(model, o, o, o, basal_scp, x, d, k, personalbias)
    end
    return model
end

# Opinion initialization function
"""
    init_opinion(model, agent, init)

Initialize the opinion of an agent in the model.

# Arguments
- `model`: The model containing the agents.
- `agent`: The agent for which the opinion is initialized.

# Returns
- `opinion`: The initialized opinion value.
"""
function init_opinion(model, index, init)
    if init == "zero"
        opinion = 0.0
    elseif init == "random"
        opinion = 0.01 * (rand(abmrng(model)) - 0.5)
    elseif init == "binary"
        opinion = rand(abmrng(model)) > 0.5 ? 1.0 : -1.0
    else
        error("Invalid opinion initialization method. Use 'zero', 'random' or 'binary'.")
    end
    return opinion
end

# Unit parameter initialization function
"""
    init_unitparam(model, index, dampingtype, dampingparam, scalingtype, scalingparam)

Initialize the damping and scaling parameters for an agent in the model.

# Arguments
- `model`: The model containing the agents.
- `index`: The index of the agent for which the parameters are initialized.
- `dampingtype`: The type of damping parameter initialization.
- `dampingparam`: The parameters for the damping initialization.
- `scalingtype`: The type of scaling parameter initialization.
- `scalingparam`: The parameters for the scaling initialization.

# Returns
- `d`: The initialized damping parameter.
- `k`: The initialized scaling parameter.
"""
function init_unitparam(model, index, dampingtype, dampingparam, scalingtype, scalingparam)
    if dampingtype == "random"
        d = randn(abmrng(model)) * dampingparam[2] + dampingparam[1]
        # clip at zero
        d = d < 0 ? 0 : d
    elseif dampingtype == "constant"
        d = dampingparam[1]
    else
        error("Invalid damping type. Use 'random' or 'constant'.")
    end
    if scalingtype == "random"
        k = randn(abmrng(model)) * scalingparam[2] + scalingparam[1]
        # clip at zero
        k = k < 0 ? 0 : k
    elseif scalingtype == "constant"
        k = scalingparam[1]
    else
        error("Invalid scaling type. Use 'random' or 'constant'.")
    end
    return d, k
end

# Generation of connectivity graphs
"""
    network_generation(jz_network, ju_network, jx_network, numagents, seed)

Generate a communication network and an susceptibility network.

# Arguments
- `jz_network`: A tuple `(budtype, budparam)` specifying the type and parameters of the communication network.
- `ju_network`: A string specifying the type of the susceptibility network.
- `jx_network`: A string specifying the type of the gating network.
- `numagents`: An integer specifying the number of agents in the network.
- `seed`: An integer specifying the seed for random number generation.

# Returns
- `g_bud`: A `LightGraphs.SimpleGraph` object representing the communication network.
- `adj_ju`: A sparse adjacency matrix representing the susceptibility network.
- `adj_jx`: A sparse adjacency matrix representing the gating network.

# Errors
- Throws an error if `budtype` is not equal to "barabasialbert".
- Throws an error if `ju_network` is not equal to "identity" or "buddies".
- Throws an error if `jx_network` is not equal to "complete".
"""
function network_generation(jz_network::Tuple, ju_network::String, jx_network::String,
                            numagents::Int, seed::Int)
    budtype, budparam = jz_network
    if budtype == "barabasialbert"
        g_bud = barabasi_albert(numagents, budparam, seed = seed)
    elseif budtype == "erdosrenyi"
        g_bud = erdos_renyi(numagents, numagents*budparam, seed = seed)
    elseif budtype == "wattsstrogatz"
        g_bud = watts_strogatz(numagents, budparam*2, 0.1, seed = seed)
    elseif budtype == "complete"
        g_bud = complete_graph(numagents)
    else
        error("Unimplemented communication network type. Use 'barabasialbert', 'erdosrenyi', 'wattsstrogatz', or 'complete'.")
    end
    if ju_network == "identity"
        adj_ju = sparse(1:numagents, 1:numagents, 1.0)
    elseif ju_network == "buddies"
        adj_ju = adjacency_matrix(g_bud)
    else
        error("Unimplemented susceptibility network type. Use 'identity' or 'buddies'.")
    end
    if jx_network == "complete"
        adj_jx = sparse(complete_graph(numagents))
    elseif jx_network == "2degree"
        # Get the 2-degree neighbors of each agent from the adjacency matrix of the social graph
        adj_jx = adjacency_matrix(g_bud)
        adj_jx = adj_jx * adj_jx 
        adj_jx = adj_jx .> 0  # Convert to binary adjacency matrix
    else
        error("Unimplemented gating network type. Use 'complete' or '2degree'.")
    end
    return g_bud, adj_ju, adj_jx
end

# Define nearby agents based on communication graph
"""
    nearby_agents(agent, model; type = "undir")

Given an agent and a model, this function returns the indices and weights of the nearby agents in the graph.

# Arguments
- `agent`: The agent for which nearby agents are to be found.
- `model`: The model containing the graph of agents.
- `type`: (optional) The type of nearby agents to find. Default is "undir".

# Returns
- `ng_idxs`: The indices of the nearby agents.
- `ng_weights`: The weights of the connections to the nearby agents.

# Raises
- `Error`: If the agent ID does not correspond to a valid vertex in the graph.
- `Error`: If an invalid type is specified.
"""
function nearby_agents(agent, model; type = "undir")
    graph = model.buddies
    adj_a = adjacency_matrix(graph)
    if !has_vertex(graph, agent.id)
        error("Agent ID does not correspond to a valid vertex in the graph.")
    end
    if type == "undir"
        agentbuddies = adj_a[agent.id, :]
        ng_idxs, ng_weights = findnz(agentbuddies)
    else
        error("Unimplemented type. Use 'undir' for undirected graph.")
    end
    return ng_idxs, ng_weights
end

# Synchronous scheduling:
# Agents have  `temp` and `new` fields for attributes that are changed via synchronous update. 
# In `agent_step!` we use the `temp` field; after updating all agents `new` fields, 
# we use the `model_step!` to update the model for the next iteration.

"""
    saturation(x; type = "tanh")

Apply the saturation function to the input `x`.

## Arguments
- `x`: The input value to apply the saturation function to.
- `type`: The type of saturation function to use. Default is "tanh".

## Returns
The result of applying the saturation function to `x`.

## Raises
- `Error`: If the specified type of saturation function is not implemented.
"""
function saturation(x; type::String = "tanh", alpha::Float64 = 1.0, beta::Float64 = 0.0)
    if type == "tanh"
        return tanh(x)
    elseif type == "sigmoid"
        return 1 / (1 + exp(-alpha*x + beta))
    else
        error("Saturation function not implemented. Use 'tanh' or 'sigmoid'.")
    end
end

"""
    bias(agent, model; type::String = "none", evol::Vector{Float64} = rand(1000))

The `bias` function calculates the internal bias for an agent for a time step.

## Arguments
- `agent`: The agent for which the bias is calculated.
- `model`: The model used to calculate the bias.
- `type::String`: The type of bias calculation.

## Returns
- The internal bias value.

## Raises
- `Error`: If the specified type of bias function is not implemented.
"""
function intbias(agent, model; type::String)
    step = abmtime(model) + 1
    if type == "none"
        return 0.0
    elseif type == "manual"
        return agent.personalbias * model.biasproc[step]
    elseif type == "randomsmall"
        return 0.01 * (rand(abmrng(model)) -1)
    elseif type == "randomlarge"
        return 0.1 * (rand(abmrng(model)) - 1)
    elseif type == "time"
        biasproc = model.biasproc
        return biasproc[step]
    else
        error("Bias function not implemented. Use 'none', 'manual', 'randomsmall', 'randomlarge', or 'time'.")
    end
end

"""
    extinput(agent, model)

The `extinput` function returns the external input for an agent at a given time step.

## Arguments
- `agent`: The agent for which the external input is calculated.
- `model`: The model used to calculate the external input.

## Returns
- The external input value.
"""
function extinput(agent, model, matrix)
    step = abmtime(model) + 1
    return matrix[agent.id, step]
end

"""
    agent_step!(agent, model)

The `agent_step!` function updates the state of an agent in a model based on its neighbors' opinions and susceptibility states. 

## Arguments
- `agent`: The agent object to update.
- `model`: The model object containing the agents and their connections.

## Description
The function performs the following steps:
1. Retrieves the indices and weights of the nearby agents using the `nearby_agents` function.
2. Updates the agent's previous opinion value.
3. Obtain the agent-specific bias term based on internal bias and external input.
4. Updates the agent's opinion by combining its self-inhibition, neighbor information, and bias term.
5. Updates the agent's susceptibility state by considering the basal susceptibility, susceptibility gain, and neighbor opinion.

## Returns
- None.
"""
function agent_step!(agent, model)
    agent.opinion_prev = agent.opinion_temp
    # Time or agent-specific bias term
    #det_input = intbias(agent, model; type = model.biastype) + extinput(agent, model)
    det_input = extinput(agent, model, model.inputmatrix)
    stoch_input = extinput(agent, model, model.noisematrix)
    # Discrete opinion update
    det_opinion_update, stoch_opinion_update = ddt_opinionstate(agent, model, det_input, stoch_input)
    opinion_alt = agent.opinion_prev + model.euler_h/2 * det_opinion_update + sqrt(model.euler_h/2) * stoch_opinion_update
    agent.opinion_new = agent.opinion_prev + model.euler_h * det_opinion_update + sqrt(model.euler_h) * stoch_opinion_update
    # Discrete susceptibility update
    scp_update = ddt_scpstate(agent, model)
    scp_alt = agent.scp_state + model.euler_h/2 * scp_update
    agent.scp_state += (model.euler_h * scp_update)
    if agent.scp_state < 0
        agent.scp_state = 0
    end
    # Discrete gate update 
    if model.gating
        det_gate_update, stoch_gate_update = ddt_gatingstate(agent, model, det_input, stoch_input)
        gate_alt = agent.gating_state + model.euler_h/2 * det_gate_update + sqrt(model.euler_h/2) * stoch_gate_update
        agent.gating_state += (model.euler_h * det_gate_update) + sqrt(model.euler_h) * stoch_gate_update
    else
        gate_alt = 0.0
    end
    # Check and warn for numerical stability of Euler approximation
    step = abmtime(model)
    if step % 100 == 0
        newstate = [agent.opinion_new, agent.scp_state, agent.gating_state]
        altstate = [opinion_alt, scp_alt, gate_alt]
        # maximum norm of the difference vector
        stability_distance = maximum(abs.(newstate - altstate))
        tolerance = 1e-3
        if stability_distance > tolerance
            println("Warning: Euler approximation of update at step $step for agent $agent may be unstable:")
            println("Maximum norm of difference vector is $stability_distance")
        end
    end
end

function ddt_opinionstate(agent, model, det_input, stoch_input)
    buddiesidxs, buddiesweights = nearby_agents(agent, model; type = "undir")
    selfinhib = -agent.damping * agent.opinion_temp
    neighborinfo = 0
    for (widx, buddyidx) in enumerate(buddiesidxs)
        buddiness = buddiesweights[widx]
        info = model[buddyidx].opinion_temp .* buddiness
        neighborinfo += info
    end
    phigate = 1.0
    if model.gating
        phigate = saturation(agent.gating_state, type = "sigmoid",
                             alpha = model.alpha_gate, beta = model.beta_gate)
    end
    det_opinion_update = phigate * (selfinhib + saturation(agent.scp_state * (neighborinfo + agent.opinion_temp), 
                                                       type = model.saturation)) + det_input
    stoch_opinion_update = stoch_input
    return det_opinion_update, stoch_opinion_update                   
end

function ddt_scpstate(agent, model)
    scpidxs, scpweights = findnz(model.adj_ju[agent.id, :])
    neighborscp = 0
    for (widx, scpidx) in enumerate(scpidxs)
        buddiness = scpweights[widx]
        scp = (model[scpidx].opinion_temp^2) .* buddiness
        neighborscp += scp
    end
    scp_update = 1/model.tau_scp * (-agent.scp_state + model.basal_scp + (model.gain_scp * neighborscp))
    return scp_update
end

function ddt_gatingstate(agent, model, det_input, stoch_input)
    gateidxs, gateweights = findnz(model.adj_jx[agent.id, :])
    addx = 0
    for (widx, gateidx) in enumerate(gateidxs)
        buddiness = gateweights[widx]
        x = saturation(model[gateidx].opinion_temp, 
                       type = model.saturation) .* buddiness
        addx += x
    end
    scaledsignal = agent.scaling * det_input
    det_gate_update = 1/model.tau_gate * (-agent.gating_state + addx + scaledsignal)
    scalednoise = agent.scaling * stoch_input
    stoch_gate_update = 1/model.tau_gate * scalednoise
    return det_gate_update, stoch_gate_update
end


"""
Update the opinions of all agents in the model.

Parameters:
- `model`: The model object containing the agents.

Returns:
- None

This function updates the `opinion_temp` attribute of each agent in the model
with the new opinion value stored in the `opinion_new` attribute.
"""
function model_step!(model)
    for a in allagents(model)
        a.opinion_temp = a.opinion_new
    end
end

# ## Running the model
# The parameter of interest is now `:opinion_new`, so we assign
# it to variable `adata` and pass it to the `run!` method
# to be collected in a DataFrame.

# In addition, we want to run the model only until all agents have converged to an opinion.
# With [`step!`](@ref) instead of specifying the amount of steps we can specify a function.
"""
    terminate(model, s; tol::Float64 = 1e-12) -> Bool

Check if the maximum number of steps has been reached or if the opinions of all agents have converged.

# Arguments
- `model`: The model containing the agents.
- `s`: Placeholder argument (ignored).
- `tol::Float64`: The tolerance for convergence. Default is 1e-12.

# Returns
- `true` if the opinions of all agents have converged or maxsteps is exceeded, `false` otherwise.
"""
function terminate(model, s; tol::Float64 = 1e-12)
    if abmtime(model) <= 2
        return false
    elseif model.maxsteps > 0 && abmtime(model) >= model.maxsteps
        return true
    elseif any(
        !isapprox(a.opinion_prev, a.opinion_new; rtol = tol)
        for a in allagents(model)
    )
        return false
    else
        return true
    end
end

# Wrap everything in a function to do some data collection using [`run!`](@ref) at every step
"""
    model_run(torecord::Vector{Symbol}; kwargs...)

Run the BFL model for a specified number of steps.

# Arguments
- `torecord::Vector{Symbol}`: The agent data to record. Default is `[:opinion_new]`.
- `kwargs...`: Additional keyword arguments.

# Returns
- `agent_data`: The agent data after running the model.
"""
function model_run(torecord = [:opinion_new]; kwargs...)
    model = init_bfl_model(; kwargs...)
    agent_data, _ = run!(model, terminate; adata = torecord)
    return agent_data
end

#using GLMakie
#using GraphMakie, GraphMakie.NetworkLayout
# Create communication graph 
# with weighted adjacency matrix adj_a with 0/1 entries
#comm_g = barabasi_albert(100, 2, seed=23)
#adj_a = adjacency_matrix(comm_g)
#fig = graphplot(comm_g; layout=Stress(; dim=3))

# Create belief system graph between options
# with weighted adjacency matrix adj_o with real entries between -1 and 1
#belief_g = erdos_renyi(num_options, 0.2, seed=24)
#adj_o = adjacency_matrix(belief_g)
#randm = 2.0 .* rand(rng, Float64, (num_options, num_options)) .- 1.0
#weights_o = adj_o .* randm
#fig = graphplot(belief_g; layout=Stress(; dim=2))