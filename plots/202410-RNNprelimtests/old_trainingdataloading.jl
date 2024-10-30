#= 
Historical code that was used to load and/or create data in the first architecture tests
=#

dataset_size = 10000
train_size = 0.8
val_size = 0.1
test_size = 0.1

# Load training data
#X = load("data/rnn_flux_data.jld2","X")
Xmask = load("data/rnn_flux_data.jld2","Xmask")
Xtrace = load("data/rnn_flux_data.jld2","Xtrace")
Y = load("data/rnn_flux_data.jld2","Y")
mean(Y)
ranks = load("data/rnn_flux_data.jld2","ranks")

# Reshape row-observations to column-observations
#X = Float32.(permutedims(X, [2,3,1]))
Xmask = permutedims(Xmask, [2,3,1])
Xtrace = permutedims(Xtrace, [2,1])
Y = permutedims(Y, [2,1])


if TASK == "small_classification"
    X_data1 = zeros(Float32, (100, 10000))
    for i in 1:10000
        if Y[i] == 1 # Low rank
            X_data1[:,i] = pad_input(Float32.(vec(gen_matrix(8,8,1, seed=10+i))),100)
        else # Full rank
            X_data1[:,i] = pad_input(Float32.(vec(gen_matrix(8,8,8, seed=10+i))),100)
        end
    end
    X = X_data1
end

if TASK == "random traces"
    X = Xtrace
end

if TASK == "small_reconstruction"
    if samerank == true  # RANK 1 ONLY
        X_data2 = zeros(Float32, (64, 10000))
        for i in 1:10000
            X_data2[:,i] = Float32.(vec(gen_matrix(8,8,1, seed=10+i)))
        end
        ranks = [1 for i in Y]
        Y = copy(X_data2)
        # Randomly replace half of each column with zeros
        for i in 1:10000
            idxs = randperm(64)[1:32]
            X_data2[idxs,i] .= 0.0
        end
        X = X_data2
    else # RANK 1 or 2
        X_data2 = zeros(Float32, (64, 10000))
        for i in 1:10000
            if Y[i] == 1    # rank 1
                X_data2[:,i] = Float32.(vec(gen_matrix(8,8,1, seed=10+i)))
            else            # rank 2
                X_data2[:,i] = Float32.(vec(gen_matrix(8,8,2, seed=10+i)))
            end
        end
        ranks = [i == 1 ? 1 : 2 for i in Y]
        Y = copy(X_data2)
        # Randomly replace half of each column with zeros
        for i in 1:10000
            idxs = randperm(64)[1:32]
            X_data2[idxs,i] .= 0.0
        end
        X = X_data2
    end
end 