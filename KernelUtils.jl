function rbf_kernel(X::Matrix, sigma::Real)
  # Compute the Gram matrix (X * X')
  G = X * X'
  # Extract the diagonal elements (squared norms of each row)
  d = diag(G)
  # Compute squared Euclidean distances using broadcasting
  D = d .+ d' .- 2G
  # Compute the RBF kernel matrix
  K = exp.(-D / (2 * sigma^2))
  return K
end

function rbf_kernel_test(X_train::Matrix, X_test::Matrix, sigma::Real)
  G_test = X_train * X_test'
  d_train = sum(X_train.^2, dims=2)
  d_test = sum(X_test.^2, dims=2)
  D = d_train .+ d_test' .- 2G_test
  return exp.(-D / (2 * sigma^2))
end

function predict_krr(X_train::Matrix, X_test::Matrix, G::AbstractMatrix, β::Vector, sigma::Real)
  K_test = rbf_kernel_test(X_train, X_test, sigma)
  return K_test' * Transpose(G) *β
end

function predict_krr(X_train::Matrix, X_test::Matrix, G::AbstractMatrix, B::Matrix, sigma::Real)
  K_test = rbf_kernel_test(X_train, X_test, sigma)
  return K_test'* Transpose(G) * B
end

function generate_nystrom_G(n::Int, m::Int; rng=Random.GLOBAL_RNG)
  row_indices = sample(1:n, m; replace=false)
  G = Matrix{Float64}(I, n, n)[row_indices, :]                       # convert to a dense matrix if needed
end