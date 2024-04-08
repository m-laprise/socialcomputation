# Julia implementation of the BFL model for multi-agent opinion formation (Leonard et al. 2023)
# Discrete time version
# by Marie-Lou Laprise
# This code was written for Julia 1.10.2 and Agents 6.0.7

using Agents
using CairoMakie
using DataFrames
using Graphs
using Random
using LinearAlgebra
using SparseArrays
using Statistics
#using DrWatson

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
- `attention_state::Float64`: The attention state of the agent.
- `damping::Float64`: The damping factor of the agent for the opinion update. By default, it will be inherited from the model and constant for all agents.
- `personalbias::Float64`: The bias term for the agent. By default, it will be inherited from the model and set to 0.
"""
@agent struct OpinionatedGuy(NoSpaceAgent)
    opinion_temp::Float64
    opinion_new::Float64
    opinion_prev::Float64
    attention_state::Float64
    damping::Float64
    personalbias::Float64
end

"""
    init_bfl_model(; numagents, damping, basal_attention, 
                    tau_attention, saturation, biastype, 
                    budnetworkparam, attnetworkparam, seed) -> model::ABM

Create an ABM model for the BFL simulation. It is a keyword function that initializes 
the model with the specified parameters, which can be used for parameter scanning.

# Arguments
- `numagents::Int`: The number of agents in the model. Default is 100.
- `damping::Float64`: The damping factor for the opinion update. 
- `basal_attention::Float64`: The basal attention state of the agents. 
- `tau_attention::Float64`: The time constant for the attention state update. 
- `saturation::String`: The saturation function for the attention state. Default is "tanh" (hyperbolic tangent for opinions between -1 and 1).
- `biastype::String`: The bias function. Default is "random" (random bias for each agent at each step).
- `budnetworkparam::Tuple`: The parameters for the communication network. Default is ("barabasialbert", 2) (Barabasi-Albert network with m=2).
- `attnetworkparam::String`: The attention network type. Default is "identity" (no attention).
- `maxsteps::Int`: The maximum number of steps to run the model. Default is 1000. 0 means run until convergence.
- `seed::Int`: The seed for the random number generator. Default is 23.

# Returns
- `model::ABM`: The created ABM model.

"""
function init_bfl_model(; numagents::Int = 100, 
                        damping::Float64 = 0.3, 
                        basal_attention::Float64=0.1, 
                        attention_gain::Float64=0.2,
                        tau_attention::Float64 = 0.05,
                        saturation::String = "tanh",
                        biastype::String = "random",
                        budnetworkparam::Tuple = ("barabasialbert", 2),
                        attnetworkparam::String = "identity",
                        maxsteps::Int = 500,
                        seed::Int=23)
    # Create an ABM model with the OpinionatedGuy agent type
    # Use the fastest scheduler for agent updates
    # Set the properties of the model
    g_bud, adj_att = network_generation(budnetworkparam, attnetworkparam, numagents, seed)
    model = StandardABM(OpinionatedGuy; agent_step!, model_step!, 
        rng = MersenneTwister(seed), scheduler = Schedulers.fastest,
        properties = Dict(
            :damping => damping,
            :basal_attention => basal_attention,
            :attention_gain => attention_gain,
            :tau_attention => tau_attention,
            :saturation => saturation,
            :biastype => biastype,
            :buddies => g_bud, # communication graph
            :adj_att => adj_att, # attention graph
            :maxsteps => maxsteps
        )
    )
    # Add numagents agents to the model
    for _ in 1:numagents
        o = rand(abmrng(model))-0.5
        # Set opinion_temp, opinion_new, opinion_prev to same random initial value
        # Set initial attention to basal, set damping
        add_agent!(model, o, o, o, basal_attention, damping, 0.0)
    end
    return model
end

# Generation communication and attention graphs
"""
    network_generation(budnetworkparam, attnetworkparam, numagents, seed)

Generate a communication network and an attention network.

# Arguments
- `budnetworkparam`: A tuple `(budtype, budparam)` specifying the type and parameters of the communication network.
- `attnetworkparam`: A string specifying the type of the attention network.
- `numagents`: An integer specifying the number of agents in the network.
- `seed`: An integer specifying the seed for random number generation.

# Returns
- `g_bud`: A `LightGraphs.SimpleGraph` object representing the communication network.
- `adj_att`: A sparse adjacency matrix representing the attention network.

# Errors
- Throws an error if `budtype` is not equal to "barabasialbert".
- Throws an error if `attnetworkparam` is not equal to "identity" or "buddies".
"""
function network_generation(budnetworkparam::Tuple, attnetworkparam::String, numagents::Int, seed::Int)
    budtype, budparam = budnetworkparam
    atttype = attnetworkparam
    if budtype == "barabasialbert"
        g_bud = barabasi_albert(numagents, budparam, seed = seed)
    elseif budtype == "erdosrenyi"
        g_bud = erdos_renyi(numagents, numagents*budparam, seed = seed)
    elseif budtype == "wattsstrogatz"
        g_bud = watts_strogatz(numagents, budparam*2, 0.1, seed = seed)
    elseif budtype == "complete"
        g_bud = complete_graph(numagents)
    else
        error("Invalid communication network type. Use 'barabasialbert', 'erdosrenyi', 'wattsstrogatz', or 'complete'.")
    end
    if atttype == "identity"
        adj_att = sparse(1:numagents, 1:numagents, 1.0)
    elseif atttype == "buddies"
        adj_att = adjacency_matrix(g_bud)
    else
        error("Invalid attention network type. Use 'identity' or 'buddies'.")
    end
    return g_bud, adj_att
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
function saturation(x; type = "tanh")
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

The `bias` function calculates the bias for an agent based on the given model.

## Arguments
- `agent`: The agent for which the bias is calculated.
- `model`: The model used to calculate the bias.
- `type::String`: The type of bias calculation. Default is "random".
- `evol::Vector{Float64}`: The evolution vector used for bias calculation. Default is a vector of 1000 random numbers.

## Returns
- The calculated bias value.

## Raises
- `Error`: If the specified type of bias function is not implemented.
"""
function bias(agent, model; type::String = "none", evol::Vector{Float64} = rand(1000))
    if type == "none"
        return 0.0
    elseif type == "random"
        return 0.1 * (rand(abmrng(model)) - 1)
    elseif type == "time"
        return evol[abmtime(model)]
    elseif type == "agenttime"
        return agent.personalbias * evol[abmtime(model)]
    else
        error("Bias function not implemented. Use 'none', 'random', 'time', or 'agenttime'.")
    end
end

"""
    agent_step!(agent, model)

The `agent_step!` function updates the state of an agent in a model based on its neighbors' opinions and attention states. 

## Arguments
- `agent`: The agent object to update.
- `model`: The model object containing the agents and their connections.

## Description
The function performs the following steps:
1. Retrieves the indices and weights of the nearby agents using the `nearby_agents` function.
2. Updates the agent's previous opinion value.
3. Calculates the agent-specific bias term based on the model's bias type.
4. Updates the agent's opinion by combining its self-inhibition, neighbor information, and bias term.
5. Updates the agent's attention state by considering the basal attention, attention gain, and neighbor opinion.

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
    agent.opinion_new = selfinhib + saturation(agent.attention_state * neighborinfo, 
                                               type = model.saturation) + abias
    # Discrete attention update
    basal_attention = model.basal_attention
    attention_gain = model.attention_gain
    tau_attention = model.tau_attention
    attidxs, attweights = findnz(model.adj_att[agent.id, :])
    neighboratt = 0
    for (widx, attidx) in enumerate(attidxs)
        buddiness = attweights[widx]
        att = (model[attidx].opinion_temp^2) .* buddiness
        neighboratt += att
    end
    agent.attention_state = tau_attention*(-agent.attention_state + basal_attention + (attention_gain * neighborinfo))
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
function model_run(torecord::Vector{Symbol} = [:opinion_new]; kwargs...)
    model = init_bfl_model(; kwargs...)
    agent_data, _ = run!(model, terminate; adata = torecord)
    return agent_data
end

#model = init_bfl_model()
#Agents.step!(model, 20)
#model[1]

data = model_run([:opinion_new]; 
                 numagents = 100, damping = 0.01, basal_attention = 0.01, tau_attention = 0.05, seed = 23)
results = DataFrame(data)
# Data has three columns: `:time`, `:id`, and `:opinion_new`.
# We can plot the opinion of all agents over time
# run, collect the data and plot it.
CairoMakie.activate!() # hide
fig = Figure(size = (600, 600))
ax = fig[1, 1] = Axis(
        fig,
        xlabel = "Step",
        ylabel = "Opinion",
        title = "Opinion formation",
    )
for grp in groupby(results, :id)
    lines!(ax, grp.time, grp.opinion_new, color = :blue, alpha = 0.5)
end
fig


# Run and plot the model for different values of the hyper parameter
const cmap = cgrad(:lightrainbow)
plotsim(ax, data) =
    for grp in groupby(data, :id)
        lines!(ax, grp.time, grp.opinion_new, color = cmap[grp.id[1]/100])
    end

xs = [0, 0.15, 0.5, 1]
figure = Figure(size = (600, 1200))
for (i, x) in enumerate(xs)
    ax = figure[i, 1] = Axis(figure; title = "x = $x")
    x_data = model_run(damping = x)
    plotsim(ax, x_data)
end
figure



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