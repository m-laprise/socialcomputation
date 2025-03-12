
#====DISTANCE BASED LOSS FUNCTIONS (Y and Yhat)====#

function MAE(A::AbstractArray{Float32, 2}, B::AbstractArray{Float32, 2}) 
    loss = Vector{Float32}(undef, size(A, 2))
    for (i, (a, b)) in enumerate(zip(eachcol(A), eachcol(B)))
        loss[i] = MAELoss(; agg = mean)(a, b)
    end
    return loss
end
MAE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = mean(abs, A .- B, dims=1)
function MSE(A::AbstractArray{Float32, 2}, B::AbstractArray{Float32, 2}) 
    loss = Vector{Float32}(undef, size(A, 2))
    for (i, (a, b)) in enumerate(zip(eachcol(A), eachcol(B)))
        loss[i] = MSELoss(; agg = mean)(a, b)
    end
    return loss
end
MSE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = mean(abs2, A .- B, dims=1)
RMSE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = sqrt.(MSE(A, B))
function MHE(A::AbstractArray{Float32, 2}, B::AbstractArray{Float32, 2}, δ::Float32 = 1f0) 
    loss = Vector{Float32}(undef, size(A, 2))
    for (i, (a, b)) in enumerate(zip(eachcol(A), eachcol(B)))
        loss[i] = HuberLoss(; delta = δ, agg = mean)(a, b)
    end
    return loss
end

"""Convenience function to dispatch to example-level loss function 
based on the dimensionality of the inputs, and return the mean over examples"""
function batchmeanloss(lossfunction::Function,
                       ys::AbstractArray{Float32, 3}, 
                       ys_hat::AbstractArray{Float32, 2};
                       datascale::Float32 = 1f0)::Float32
    mean(lossfunction(_2D(ys), ys_hat)) / datascale
end
function batchmeanloss(lossfunction::Function,
                       ys::AbstractArray{Float32, 2}, 
                       ys_hat::AbstractArray{Float32, 2};
                       datascale::Float32 = 1f0)::Float32
    mean(lossfunction(ys, ys_hat)) / datascale
end

"""Convenience function to dispatch to example-level loss function 
based on the dimensionality of the inputs, and return the vector of losses 
for each example"""
function batchvecloss(lossfunction::Function,
                       ys::AbstractArray{Float32, 3}, 
                       ys_hat::AbstractArray{Float32, 2};
                       datascale::Float32 = 1f0)
    (lossfunction(_2D(ys), ys_hat)) / datascale
end
function batchvecloss(lossfunction::Function,
                       ys::AbstractArray{Float32, 2}, 
                       ys_hat::AbstractArray{Float32, 2};
                       datascale::Float32 = 1f0)
    (lossfunction(ys, ys_hat)) / datascale
end

"""Take 2D (N^2 x nb_examples) or 3D arrays (N x N x nb_examples) with all entries 
and return dense 2D arrays of known entries (KNOWNENTRIES x nb_examples)"""
function reducetoknown(ys::AbstractArray{Float32, 3}, 
                       ys_hat::AbstractArray{Float32, 2},
                       nonzeroidx::AbstractVector{<:Real})
    _2D(ys)[nonzeroidx, :], ys_hat[nonzeroidx, :]
end
function reducetoknown(ys::AbstractArray{Float32, 2}, 
                       ys_hat::AbstractArray{Float32, 2},
                       nonzeroidx::AbstractVector{<:Real})
    ys[nonzeroidx, :], ys_hat[nonzeroidx, :]
end

"Apply arbitrary loss function to known entries only"
function meanknownentriesloss(lossfunction::Function,
                          ys::AbstractArray{Float32, 3}, 
                          ys_hat::AbstractArray{Float32, 2},
                          nonzeroidx::AbstractVector{<:Real};
                          datascale::Float32 = 1f0)::Float32
    ys_known, ys_hat_known = reducetoknown(ys, ys_hat, nonzeroidx)
    batchmeanloss(lossfunction, ys_known, ys_hat_known, datascale = datascale)
end
function vecknownentriesloss(lossfunction::Function,
                          ys::AbstractArray{Float32, 3}, 
                          ys_hat::AbstractArray{Float32, 2},
                          nonzeroidx::AbstractVector{<:Real};
                          datascale::Float32 = 1f0)
    ys_known, ys_hat_known = reducetoknown(ys, ys_hat, nonzeroidx)
    batchvecloss(lossfunction, ys_known, ys_hat_known, datascale = datascale)
end

#====STRUCTURE BASED LOSS FUNCTIONS (Yhat only)====#

# SVD decomposition, when running simulations on millions on matrices,
# can sometimes (rarely) generate LAPACK errors due to numerical instability. This can be avoided
# by changing the algorithm used when an error occurs; this requires using svd(A).S instead of svdvals(A)
# and a try catch (credit to https://yanagimotor.github.io/posts/2021/06/blog-post-lapack/ for the tip):
#=function robust_svdvals(A::AbstractArray{Float32, 2})
    try
        return svdvals(A)
    catch e
        @warn "LAPACK error detected; switching to QR iteration"
        return svd(A, alg=LinearAlgebra.QRIteration()).S
    end
end=#
# The fallback is much slower, but avoids the error and is rarely used.
# HOWEVER; I can't autodiff through the QRIteration. Should file an issue on Github.
# In the meantime, I simply skip the penalty when this happens.



"Nuclear norm of a matrix (sum of singular values)"
nuclearnorm(A::AbstractArray{Float32, 2})::Float32 = sum(svdvals(A))

"Nuclear norm of a matrix (sum of singular values), divided by the number of singular values"
meansvdvals(A::AbstractArray{Float32, 2})::Float32 = sum(svdvals(A)) / size(A, 1)

# Nuclear norm, but scaled to avoid minimization resulting in the matrix entries going to zero.
"Nuclear norm of a matrix (sum of singular values), scaled by the standard deviation of its entries"
scalednuclearnorm(A::AbstractArray{Float32, 2})::Float32 = sum(svdvals(A)) / (size(A, 1) * std(A))
scalednuclearnorm(svdvals::AbstractVector{Float32}, scaling::Float32)::Float32 = sum(svdvals) / (length(svdvals) * scaling)

"Spectral norm of a matrix (largest singular value)"
spectralnorm(A::AbstractArray{Float32, 2})::Float32 = svdvals(A)[1]

"Spectral gap of a matrix (largest singular value - second largest singular value)"
function spectralgap(A::AbstractArray{Float32, 2})::Float32
    vals = svdvals(A)
    return vals[1] - vals[2] 
end

# Spectral gap, but scaled to avoid maximization resulting in the first singular value getting arbitrarily large
"Spectral gap of a matrix (largest singular value - second largest singular value), 
divided by the largest singular value"
function scaledspectralgap(A::AbstractArray{Float32, 2})::Float32
    vals = svdvals(A)
    return (vals[1] - vals[2]) / vals[1] 
end

# The ground truth absolute nuclear norms vary between 100 and 30, with a mean of 64.
# Divided by the number of singular values, this gives a mean of 1; 
# We can use this scale so that the penalty is order 1, making it active only for scaled nuclear norms above 1.


# Minimize the sum of all singular values other than the first
# Maximize the ratio of the first singular value to the sum of all other singular values
# If the first singular value is larger than 100, minimize it until it reaches 100
"Populates a vector with the scaled spectral gap of each matrix in a 3D array"
function populatepenalties!(penalties, ys_hat::AbstractArray{Float32, 3})::Nothing
    @inbounds for i in axes(ys_hat, 3)
        try
            valsY = svdvals(@view ys_hat[:,:,i])
            sumvals = sum(valsY[2:end])
            penalties[i] = sumvals/64f0 + sumvals/valsY[1]
            if valsY[1] > 100f0
                penalties[i] += valsY[1]/64f0
            end
            #nn = sum(valsY)
            #snn = (nn / length(valsY)) - 1f0
            #sgap = ((valsY[1] - valsY[2]) / valsY[1]) - 1f0 #Between -1 and 0; 0 is ideal
            #penalties[i] = snn >= 1f0 ? snn-sgap : 1f0-sgap
            #penalties[i] = snn >= 0.5f0 ? snn-sgap : 0f0-sgap
            #penalties[i] += snn >= 0f0 ? snn/10f0 : 0f0
            #vals1max = valsY[1] <= 60f0 ? valsY[1]/64f0 - 60f0/64f0 : 0f0 #Between -1 and 0; 0 is ideal
            #penalties[i] -= vals1max
        catch e
            @warn "LAPACK error detected; skipping spectral penalty. Error: $e"
            penalties[i] = 0f0
        end
        #penalties[i] = -scaledspectralgap(@view ys_hat[:,:,i]) + 1f0 
    end
end

#===== COMPOSITE TRAINING LOSSES =====#


# Training loss - Many data points
""" 
    Training loss given a 3D array of true matrices, a matrix where each row is a vectorized 
    predicted matrix, and the mask matrix with information about which entries are known.
    The loss is a weighted sum of the L1 loss on known entries and a scaled spectral gap penalty.
"""
function spectrum_penalized_l2(ys::AbstractArray{Float32, 3}, 
                          ys_hat::AbstractArray{Float32, 2}, 
                          nonzeroidx::AbstractVector{<:Real};
                          theta::Float32 = THETA,
                          datascale::Float32 = 0.1f0)::Float32
    nb_examples = size(ys, 3)
    # L2 loss on known entries (vector, one loss per example)
    l2_known = vecknownentriesloss(MSE, ys, ys_hat, nonzeroidx, datascale = datascale)
    # Spectral norm penalty
    penalties = Array{Float32}(undef, nb_examples)
    populatepenalties!(penalties, _3D(ys_hat))
    # Training loss
    errors = theta * l2_known .+ (1f0 - theta) * penalties 
    # Return the mean over examples
    return mean(errors)
end

function spectrum_penalized_huber(ys::AbstractArray{Float32, 3}, 
                          ys_hat::AbstractArray{Float32, 2}, 
                          nonzeroidx::AbstractVector{<:Real};
                          theta::Float32 = THETA,
                          datascale::Float32 = 1f0)::Float32
    nb_examples = size(ys, 3)
    # Huber loss on known entries (vector, one loss per example)
    hub_known = vecknownentriesloss(MHE, ys, ys_hat, nonzeroidx, datascale = datascale)
    # Spectral norm penalty
    penalties = Array{Float32}(undef, nb_examples)
    populatepenalties!(penalties, _3D(ys_hat))
    # Training loss
    errors = theta * hub_known .+ (1f0 - theta) * penalties 
    # Return the mean over examples
    return mean(errors)
end
