# Objectives of the three kernel problems in the Nyström form
#     f(B) = loss(y, F B) + (λ/2) tr(B'SB),   F = K G' (n×m),  S = G K G' (m×m).
# B is m×1 for quantile and logistic regression and m×d for multinomial
# regression, with d = q-1 (standard parameterization, last class is the
# reference) or d = q (full parameterization). These functions are used by the
# baseline methods (Newton, FISTA, AdaGD, L-BFGS) and to check every solution
# independently of the solver that produced it.

const TAU = 0.5      # quantile level
const H_BW = 0.25    # bandwidth of the Gaussian smoothing of the check loss

# Loss at η = FB; fills the residual ∂loss/∂η and the curvature weights w.
function loss!(res, w, problem, η, y; full=false)
    f = 0.0
    if problem == "quantile"
        nh = Normal(0, H_BW)
        @inbounds for i in eachindex(η)
            r = y[i] - η[i]; c = cdf(nh, -r)
            f += H_BW * exp(-(r / H_BW)^2 / 2) / sqrt(2π) + r * (TAU - c)
            res[i] = c - TAU; w[i] = pdf(nh, r)
        end
    elseif problem == "logistic"
        @inbounds for i in eachindex(η)
            v = η[i]; p = 1 / (1 + exp(-v))
            f += max(v, 0.0) + log1p(exp(-abs(v))) - y[i] * v
            res[i] = p - y[i]; w[i] = p * (1 - p)
        end
    else
        d = size(η, 2)
        @inbounds for i in axes(η, 1)
            a = full ? -Inf : 0.0            # the reference class has η = 0
            for k = 1:d; a = max(a, η[i, k]); end
            den = full ? 0.0 : exp(-a)
            for k = 1:d; den += exp(η[i, k] - a); end
            f += a + log(den)
            for k = 1:d
                w[i, k] = exp(η[i, k] - a) / den
                f -= y[i, k] * η[i, k]; res[i, k] = w[i, k] - y[i, k]
            end
        end
    end
    f
end

"""
    objective(problem, F, S, y, λ, B; full=false, gradient=true) -> (f, ∇f, w)

`problem` is "quantile", "logistic" or "multinomial"; `y` is a vector, or the
n×q one-hot matrix for multinomial regression. `w` holds the curvature weights
used by Newton's method: φ_h(r_i) (quantile), p_i(1-p_i) (logistic) or the
class probabilities p_ik (multinomial). With `gradient=false` the gradient is
not formed (∇f is `nothing`).
"""
function objective(problem, F, S, y, λ, B; full=false, gradient=true)
    η = mult(F, B); SB = mult(S, B); res = similar(η); w = similar(η)
    f = loss!(res, w, problem, η, y; full=full) + λ / 2 * dot(B, SB)
    gradient || return f, nothing, w
    f, mult(F', res) .+ λ .* SB, w
end

# A*B, as a matrix-vector product when B has one column (BLAS gemv is about
# twice as fast as gemm with one column).
mult(A, B) = size(B, 2) == 1 ? reshape(A * vec(B), :, 1) : A * B

"""Validation metric of a fitted B on new data: mean absolute deviation from the
true signal (quantile; `y` is the signal) or log-likelihood (logistic, multinomial)."""
function validation(problem, B, F, y; full=false)
    η = mult(F, B)
    problem == "quantile" && return mean(abs.(y .- η))
    -loss!(similar(η), similar(η), problem, η, y; full=full)
end

"""
    hessian(problem, F, S, λ, w; ridge)

Dense Hessian at the weights `w` returned by `objective`, as in the paper's
Newton code: F'WF + λS + ridge·I for quantile and logistic regression, and for
multinomial regression the blocks F'diag(δ_kl p_k - p_k p_l)F with λ(S + ridge·I)
added to the diagonal blocks.
"""
function hessian(problem, F, S, λ, w; ridge)
    problem == "multinomial" || return F' * (w .* F) .+ λ .* S + ridge * I
    m = size(F, 2); d = size(w, 2); H = zeros(m * d, m * d)
    for k = 1:d, l = 1:k
        c = (k == l ? w[:, k] : zeros(size(w, 1))) .- w[:, k] .* w[:, l]
        blk = F' * (c .* F)
        k == l && (blk .+= λ .* (S + ridge * I))
        H[(k-1)m+1:k*m, (l-1)m+1:l*m] = blk
        k == l || (H[(l-1)m+1:l*m, (k-1)m+1:k*m] = blk')
    end
    H
end
