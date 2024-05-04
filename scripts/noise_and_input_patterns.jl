using LinearAlgebra
using Random
using Distributions

function bias_process(alpha::Float64, start::Int, stop::Int, maxsteps::Int)
    biasproc = zeros(maxsteps)
    biasproc[start:stop] .= alpha
    return biasproc
end

function create_extinput(n::Int, maxsteps::Int; 
                         gaussiannoise::Bool = false,
                         noise_mean::Float64 = 0.0,
                         noise_var::Float64 = 0.1,
                         prestimulus_delay::Int = 50,
                         betweensignals_delay::Int = 0,
                         betweenbursts_delay::Int = 0,
                         group1::Float64 = 0.0,
                         signal1_nbbursts::Int = 1,
                         signal1_strength::String = "low",
                         signal1_timeon::Int = 10,
                         group2::Float64 = 0.0,
                         signal2_nbbursts::Int = 1,
                         signal2_strength::String = "none",
                         signal2_timeon::Int = 10,
                         mutuallyexclusivegroups::Bool = true)
    numagents_g1 = Int(round(group1 * n))
    numagents_g2 = Int(round(group2 * n))
    # ERRORS AND WARNINGS
    if numagents_g1 > n || numagents_g2 > n
        error("The number of agents in a group cannot exceed the total number of agents.")
    end
    if numagents_g1 < 0 || numagents_g2 < 0
        error("The number of agents in a group must be positive.")
    end
    if prestimulus_delay + signal1_nbbursts * (signal1_timeon + betweenbursts_delay) + betweensignals_delay + signal2_nbbursts * (signal2_timeon + betweenbursts_delay) > maxsteps
        @warn("The sum of all delays exceeds the maximum number of steps.")
    end
    # Sample agents into group
    if mutuallyexclusivegroups
        if numagents_g1 + numagents_g2 > n
            error("The number of agents in group 1 and group 2 cannot exceed the total number of agents.")
        end
        randomized_agentlist = randperm(n)
        g1_idx = randomized_agentlist[1:numagents_g1]
        g2_idx = randomized_agentlist[numagents_g1 + 1:numagents_g1 + numagents_g2]
    else
        g1_idx = randperm(n)[1:numagents_g1]
        g2_idx = randperm(n)[1:numagents_g2]
    end
    # The external input is a matrix of size (n, maxsteps)
    # Each row corresponds to the external input for one agent
    # Each column corresponds to the external input at one time step
    extinput = zeros(n, maxsteps)
    # Add Gaussian noise to the external input
    if gaussiannoise
        noise = rand(Normal(noise_mean, noise_var), n, maxsteps)
        extinput .= noise
    end
    # Add the first signal
    if group1 > 0 && signal1_strength != "none"
        if signal1_strength == "low"
            signal1 = 0.1
        elseif signal1_strength == "medium"
            signal1 = 0.25
        elseif signal1_strength == "high"
            signal1 = 0.4
        else
            error("Invalid signal strength. Choose from 'low', 'medium', 'high', or 'none'.")
        end
        for i in 1:signal1_nbbursts
            start_time1 = prestimulus_delay + (i - 1) * (signal1_timeon + betweenbursts_delay)
            end_time1 = start_time1 + signal1_timeon
            extinput[g1_idx, start_time1 + 1:end_time1] .+= signal1
        end
    end
    # Add the second signal, taking into account betweensignals_delay
    if group2 > 0 && signal2_strength != "none"
        if signal2_strength == "low"
            signal2 = 0.1
        elseif signal2_strength == "medium"
            signal2 = 0.25
        elseif signal2_strength == "high"
            signal2 = 0.4
        else
            error("Invalid signal strength. Choose from 'low', 'medium', 'high', or 'none'.")
        end
        for i in 1:signal2_nbbursts
            start_time2 = prestimulus_delay + betweensignals_delay + (i - 1) * (signal2_timeon + betweenbursts_delay)
            end_time2 = start_time2 + signal2_timeon
            extinput[g2_idx, start_time2 + 1:end_time2] .+= signal2
        end
    end
    return extinput
end

using Test

# Create unit test for the function above
function test_create_extinput()
    Random.seed!(1234)
    extinput = create_extinput(10, 100)
    @test size(extinput) == (10, 100)
    @test all(extinput .>= 0)
    @test all(extinput .<= 0.4)
end
test_create_extinput()

# test for the function above, for different values of the parameters
#Random.seed!(1234)
_extinput0 = create_extinput(10, 50, prestimulus_delay = 10)
_extinput1 = create_extinput(10, 50, gaussiannoise = true, noise_mean = 0.0, noise_var = 0.2, prestimulus_delay = 10)
_extinput2 = create_extinput(10, 50, group1 = 0.5, signal1_nbbursts = 2, betweenbursts_delay = 2,
                              signal1_strength = "medium", signal1_timeon = 2, prestimulus_delay = 10)
_extinput3 = create_extinput(10, 50, group1 = 0.5, signal1_nbbursts = 2, signal1_strength = "medium", signal1_timeon = 20, 
                                    group2 = 0.3, signal2_nbbursts = 1, signal2_strength = "high", signal2_timeon = 10, prestimulus_delay = 10)                                   
_extinput4 = create_extinput(10, 50, mutuallyexclusivegroups = false, group1 = 0.5, group2 = 0.5, signal1_strength = "low", signal2_strength = "high", 
                            prestimulus_delay = 10)
_extinput5 = create_extinput(10, 50, mutuallyexclusivegroups = false, group1 = 0.5, group2 = 0.5, betweensignals_delay = 3,
                            signal1_strength = "low", signal2_strength = "high", 
                            prestimulus_delay = 10)