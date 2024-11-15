#=
The script includes helper functions for inspecting and diagnosing gradients during training
and for analyzing the training process.
=#

#### TRAINING UTILITIES
using ForwardDiff

"""
    inspect_gradients(grads)

Inspect the gradients for NaN, vanishing, or exploding values during the training loop
and return the number of parameters with these issues.
"""
function inspect_gradients(grads)
    g, _ = Flux.destructure(grads)
    
    nan_params = [0]
    vanishing_params = [0]
    exploding_params = [0]

    if any(isnan.(g))
        push!(nan_params, 1)
    end
    if any(abs.(g) .< 1e-6)
        push!(vanishing_params, 1)
    end
    if any(abs.(g) .> 1e6)
        push!(exploding_params, 1)
    end
    return sum(nan_params), sum(vanishing_params), sum(exploding_params)
end

"""
    diagnose_gradients(n, v, e)

Takes as input the output of `inspect_gradients` and prints a message based on the number of
parameters with NaN, vanishing, or exploding gradients.
"""
function diagnose_gradients(n, v, e)
    if n > 0
        println(n, " NaN gradient detected")
    end
    if v > 0
        println(v, " vanishing gradient detected")
    end
    if e > 0
        println(e, " exploding gradient detected")
    end
    #Otherwise, report that no issues were found
    if n == 0 && v == 0 && e == 0
        println("Gradients appear to be well-behaved.")
    end
end

"""
    getjacobian(activemodel; x=nothing, wrt="state")

Compute the Jacobian of a model with respect to its input or state.

# Arguments
- `activemodel`: The model to compute the Jacobian for.
- `x`: The input to the model. Required if `wrt="input"`.
- `wrt`: The type of Jacobian to compute. Can be either "input" or "state".

# Returns
- The Jacobian matrix. Its size is `(output_size, input_size)` if `wrt="input"`, or
  `(state_size, state_size)` if `wrt="state"`.
"""
function getjacobian(activemodel, infosize::Int; x=nothing, wrt="state")
    if wrt == "input"
        @assert !isnothing(x)
        J = Zygote.jacobian(activemodel, x)[1]
    elseif wrt == "state"
        x = zeros(Float32, infosize)
        J = Zygote.jacobian(x) do y
            activemodel(y); state(activemodel.layers[1])
        end
    else
        return @warn("Invalid wrt argument. Choose from 'input' or 'state'.")
    end
    return J[1]
end

function numerical_jacobian(activemodel, infosize::Int; x=nothing, wrt="state", epsilon=1e-8)
    if wrt == "state"
        h = state(activemodel.layers[1])
        x = zeros(Float32, infosize)
        statesize = length(h)
        fx = activemodel(x)
        outputsize = length(fx)
        J = zeros(outputsize, statesize)
        reset!(activemodel.layers[1])
        for i in 1:statesize
            h_perturbed = copy(h)
            h_perturbed[i] += epsilon
            activemodel.layers[1].state = h_perturbed
            J[:, i] = (activemodel(h_perturbed) - fx) / epsilon
            reset!(activemodel.layers[1])
        end
        return J
    else
        return @warn("Invalid wrt argument.")
    end
end

reset!(activemodel.layers[1])
h = state(activemodel.layers[1])
x = zeros(Float32, 150)
n = length(h)
J = zeros(n,n)
fx = activemodel(x)
reset!(activemodel.layers[1])
h_perturbed = copy(h)
h_perturbed[1] += 1e-8
activemodel.layers[1].state = h_perturbed

J[:, 1] = (activemodel(h_perturbed) - fx) / 1e-8
reset!(activemodel.layers[1])

function test_jacobian(J1, model, tol = 1e-6)
    fwdJ = ForwardDiff.jacobian(model, x)
    diff1 = norm(J1 - fwdJ)
    if diff1 < tol
        println("Jacobian computation agrees with forward mode.")
    else
        println("Jacobian computation differs from forward mode. Norm of difference: ", round(diff, digits=6))
    end
    numJ = numerical_jacobian(model)
    diff2 = norm(J1 - numJ)
    if diff2 < tol
        println("Jacobian computation agrees with numerical approximation.")
    else
        println("Jacobian computation differs from numerical approximation. Norm of difference: ", round(diff, digits=6))
    end
end
