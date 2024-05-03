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
- `susceptibility_state::Float64`: The susceptibility state of the agent.
- `damping::Float64`: The damping factor of the agent for the opinion update. By default, it will be inherited from the model and constant for all agents.
- `personalbias::Float64`: The bias term for the agent. By default, it will be inherited from the model and set to 0.
"""
@agent struct OpinionatedGuy(NoSpaceAgent)
    opinion_temp::Float64
    opinion_new::Float64
    opinion_prev::Float64
    susceptibility_state::Float64
    damping::Float64
    personalbias::Float64
end

"""
    init_bfl_model(; numagents, damping, basal_susceptibility, susceptibility_gain,
                    tau_susceptibility, saturation, biastype, 
                    jz_network, ju_network, opinioninit, maxsteps, seed) -> model::ABM

Create an ABM model for the BFL simulation. It is a keyword function that initializes 
the model with the specified parameters, which can be used for parameter scanning.

# Arguments
- `numagents::Int`: The number of agents in the model. Default is 100.
- `damping::Float64`: The damping factor for the opinion update. 
- `basal_susceptibility::Float64`: The basal susceptibility state of the agents. 
- `tau_susceptibility::Float64`: The time constant for the susceptibility state update. 
- `saturation::String`: The saturation function for the susceptibility state. Default is "tanh" (hyperbolic tangent for opinions between -1 and 1).
- `biastype::String`: The bias function. Default is "randomsmall" (small random bias for each agent at each step).
- `biasproc::Vector{Float64}`: The evolution vector for the bias calculation if biastype = "manual". Default is a vector of random numbers.
- `numsensing::Int`: The number of agents that are sensitive to external information. Default is 0.
- `jz_network::Tuple`: The parameters for the communication network. Default is ("barabasialbert", 2) (Barabasi-Albert network with m=2).
- `ju_network::String`: The susceptibility network type. Default is "identity" (no susceptibility).
- `opinioninit::String`: The opinion initialization method. Default is "random" (random opinion between -0.5 and 0.5).
- `maxsteps::Int`: The maximum number of steps to run the model. Default is 500. 0 means run until convergence.
- `euler_h::Float64`: The Euler integration step size for the opinion and susceptibility state updates. Default is 0.1.
- `seed::Int`: The seed for the random number generator. Default is 23.

# Returns
- `model::ABM`: The created ABM model.
"""
function init_bfl_model(; numagents::Int = 100,
                        dampingtype::String = "constant", 
                        dampingparam::Tuple = (0.5, 0.25),
                        basal_susceptibility::Float64 = 0.01, 
                        susceptibility_gain::Float64 = 1.0,
                        tau_susceptibility::Float64 = 1.0,
                        saturation::String = "tanh",
                        jz_network::Tuple = ("barabasialbert", 3),
                        ju_network::String = "identity",
                        # External input
                        biastype::String = "randomsmall",
                        biasproc::Vector{Float64} = rand(model.maxsteps),
                        numsensing::Int = 0,
                        # Gate parameters
                        gating::Bool = false,
                        jx_network::String = "complete",
                        tau_gate::Float64 = 1.0,
                        alpha_gate::Float64 = 1.0,
                        beta_gate::Float64 = 0.0,
                        scalingtype::String = "gaussian",
                        scalingparam::Tuple = (1.0,0.5),
                        # Initialization and simulation
                        opinioninit::String = "random",
                        maxsteps::Int = 500,
                        euler_h::Float64 = 0.1,
                        seed::Int=23)
    # Create an ABM model with the OpinionatedGuy agent type
    # Use the fastest scheduler for agent updates
    # Set the properties of the model
    buddies, adj_ju, adj_jx = network_generation(jz_network, ju_network, jx_network, numagents, seed)
    model = StandardABM(OpinionatedGuy; agent_step!, model_step!, 
        rng = MersenneTwister(seed), scheduler = Schedulers.fastest,
        properties = Dict(
            :basal_susceptibility => basal_susceptibility,
            :susceptibility_gain => susceptibility_gain,
            :tau_susceptibility => tau_susceptibility,
            :saturation => saturation,
            :biastype => biastype,
            :biasproc => biasproc,
            :buddies => buddies, # communication graph
            :adj_ju => adj_ju, # susceptibility graph
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
    sensors = randperm(numagents)[1:numsensing]
    sensors_odd = sensors[1:2:end]
    sensors_even = sensors[2:2:end]
    # Add numagents agents to the model
    for i in 1:numagents
        o = init_opinion(model, i, opinioninit)
        if biastype == "manual"
            if i in sensors_even
                personalbias = 1.0
            elseif i in sensors_odd
                personalbias = -1.0
            else
                personalbias = 0.0
            end
        else
            personalbias = 0.0
        end
        # Set opinion_temp, opinion_new, opinion_prev to same random initial value
        # Set initial susceptibility to basal, set damping
        if dampingtype == "constant"
            damping = dampingparam[1]
        elseif dampingtype == "random"
            damping = dampingparam[1] + dampingparam[2] * (rand(model.rng) - 0.5)
        else
            error("Invalid damping type. Use 'constant' or 'random'.")
        end
        add_agent!(model, o, o, o, basal_susceptibility, damping, personalbias)
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
        opinion = 0.1 * (rand(abmrng(model)) - 0.5)
    elseif init == "binary"
        opinion = rand(abmrng(model)) > 0.5 ? 1.0 : -1.0
    else
        error("Invalid opinion initialization method. Use 'zero', 'random' or 'binary'.")
    end
    return opinion
end


# Generation communication and susceptibility graphs
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
    else
        error("Unimplemented gating network type. Use 'complete'.")
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
        error("Agent ID does not correspond to a valid vertex in the graph")
    end
    if type == "undir"
        agentbuddies = adj_a[agent.id, :]
        ng_idxs, ng_weights = findnz(agentbuddies)
    else
        error("Invalid type.")
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
function saturation(x; type::String = "tanh")
    if type == "tanh"
        return tanh(x)
    elseif type == "sigmoid"
        return 1 / (1 + exp(-x))
    else
        error("Saturation function not implemented. Use 'tanh' or 'sigmoid'.")
    end
end

"""
    bias(agent, model; type::String = "none", evol::Vector{Float64} = rand(1000))

The `bias` function calculates the bias for an agent for a time step based on the given model.

## Arguments
- `agent`: The agent for which the bias is calculated.
- `model`: The model used to calculate the bias.
- `type::String`: The type of bias calculation.

## Returns
- The calculated bias value.

## Raises
- `Error`: If the specified type of bias function is not implemented.
"""
function bias(agent, model; type::String)
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
    agent_step!(agent, model)

The `agent_step!` function updates the state of an agent in a model based on its neighbors' opinions and susceptibility states. 

## Arguments
- `agent`: The agent object to update.
- `model`: The model object containing the agents and their connections.

## Description
The function performs the following steps:
1. Retrieves the indices and weights of the nearby agents using the `nearby_agents` function.
2. Updates the agent's previous opinion value.
3. Calculates the agent-specific bias term based on the model's bias type.
4. Updates the agent's opinion by combining its self-inhibition, neighbor information, and bias term.
5. Updates the agent's susceptibility state by considering the basal susceptibility, susceptibility gain, and neighbor opinion.

## Returns
- None.
"""
function agent_step!(agent, model)
    buddiesidxs, buddiesweights = nearby_agents(agent, model; type = "undir")
    agent.opinion_prev = agent.opinion_temp
    # Time or agent-specific bias term
    abias = bias(agent, model; type = model.biastype)
    # Discrete opinion update
    selfinhib = -agent.damping * agent.opinion_temp
    neighborinfo = 0
    for (widx, buddyidx) in enumerate(buddiesidxs)
        buddiness = buddiesweights[widx]
        info = model[buddyidx].opinion_temp .* buddiness
        neighborinfo += info
    end
    opinion_update = selfinhib + saturation(agent.susceptibility_state * neighborinfo, 
                                            type = model.saturation) + abias
    agent.opinion_new = agent.opinion_prev + model.euler_h*opinion_update
    # Discrete susceptibility update
    basal_susceptibility = model.basal_susceptibility
    susceptibility_gain = model.susceptibility_gain
    tau_susceptibility = model.tau_susceptibility
    attidxs, attweights = findnz(model.adj_ju[agent.id, :])
    neighboratt = 0
    for (widx, attidx) in enumerate(attidxs)
        buddiness = attweights[widx]
        att = (model[attidx].opinion_temp^2) .* buddiness
        neighboratt += att
    end
    susceptibility_update = tau_susceptibility*(-agent.susceptibility_state + basal_susceptibility + (susceptibility_gain * neighborinfo))
    agent.susceptibility_state += model.euler_h*susceptibility_update
    if agent.susceptibility_state < 0
        agent.susceptibility_state = 0
    end
    # Check and warn for numerical stability of Euler approximation
    step = abmtime(model)
    if abs((model.euler_h * susceptibility_update) + 1) > 2
        println("Warning: Euler approximation of susceptibility update at step $step may be unstable.")
    end
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
- `true` if the opinions of all agents have converged, `false` otherwise.
"""
function terminate(model, s; tol::Float64 = 1e-12)
    if abmtime(model) <= 2
        return false
    end
    if model.maxsteps > 0 && abmtime(model) >= model.maxsteps
        return true
    end
    if any(
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

function bias_process(alpha::Float64, start::Int, stop::Int, maxsteps::Int)
    biasproc = zeros(maxsteps)
    biasproc[start:stop] .= alpha
    return biasproc
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