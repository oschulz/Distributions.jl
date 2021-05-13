
##### Generic methods #####

"""
    size(d::ArrayDistribution)

Return the size of each sample from distribution `d`.
"""
Base.size(d::ArrayDistribution)

size(d::ArrayDistribution, i) = size(d)[i]

"""
    length(d::ArrayDistribution)

The length (*i.e* number of elements) of each sample from the distribution `d`.
"""
Base.length(d::ArrayDistribution)

"""
    rank(d::ArrayDistribution)

The rank of each sample from the distribution `d`.
"""
LinearAlgebra.rank(d::ArrayDistribution)

"""
    vec(d::ArrayDistribution)

If known, returns a `MultivariateDistribution` instance representing the
distribution of vec(X), where X is a random matrix with distribution `d`.
"""
Base.vec(d::ArrayDistribution)

"""
    inv(d::MatrixDistribution)

If known, returns a `MatrixDistribution` instance representing the
distribution of inv(X), where X is a random matrix with distribution `d`.
"""
Base.inv(d::MatrixDistribution)

"""
    mean(d::ArrayDistribution)

Return the mean matrix of `d`.
"""
mean(d::ArrayDistribution)

"""
    var(d::ArrayDistribution)

Compute the matrix of element-wise variances for distribution `d`.
"""
var(d::ArrayDistribution)

var(d::MatrixDistribution) = ((n, p) = size(d); [var(d, i, j) for i in 1:n, j in 1:p])

"""
    cov(d::ArrayDistribution)

Compute the covariance matrix for `vec(X)`, where `X` is a random matrix with distribution `d`.
"""
cov(d::ArrayDistribution)

function cov(d::MatrixDistribution, ::Val{true}=Val(true))
    M = length(d)
    V = zeros(partype(d), M, M)
    iter = CartesianIndices(size(d))
    for el1 = 1:M
        for el2 = 1:el1
            i, j = Tuple(iter[el1])
            k, l = Tuple(iter[el2])
            V[el1, el2] = cov(d, i, j, k, l)
        end
    end
    return V + tril(V, -1)'
end

"""
    cov(d::ArrayDistribution{S,N}, flattened = Val(false))

Compute the `2N`-dimensional array whose `(i1, j1,..., i2,l2,...)` element is `cov(X[i1,j1,...], X[i2,j2,...])`.
"""
cov(d::ArrayDistribution, ::Val{false})

function cov(d::MatrixDistribution, ::Val{false})
    n, p = size(d)
    [cov(d, i, j, k, l) for i in 1:n, j in 1:p, k in 1:n, l in 1:p]
end

"""
    _rand!(::AbstractRNG, ::ArrayDistribution, A::AbstractMatrix)

Sample the matrix distribution and store the result in `A`.
Must be implemented by matrix-variate distributions.
"""
_rand!(::AbstractRNG, ::ArrayDistribution, A::AbstractMatrix)

## sampling

# multivariate with pre-allocated 3D array
function _rand!(rng::AbstractRNG, s::Sampleable{Matrixvariate},
                m::AbstractArray{<:Real, 3})
    @boundscheck (size(m, 1), size(m, 2)) == (size(s, 1), size(s, 2)) ||
        throw(DimensionMismatch("Output size inconsistent with matrix size."))
    smp = sampler(s)
    for i in Base.OneTo(size(m,3))
        _rand!(rng, smp, view(m,:,:,i))
    end
    return m
end

# multiple matrix-variates with pre-allocated array of maybe pre-allocated matrices
rand!(rng::AbstractRNG, s::Sampleable{Matrixvariate},
      X::AbstractArray{<:AbstractMatrix}) =
          @inbounds rand!(rng, s, X,
                          !all([isassigned(X,i) for i in eachindex(X)]) ||
                          (sz = size(s); !all(size(x) == sz for x in X)))

function rand!(rng::AbstractRNG, s::Sampleable{Matrixvariate},
               X::AbstractArray{M}, allocate::Bool) where M <: AbstractMatrix
    smp = sampler(s)
    if allocate
        for i in eachindex(X)
            X[i] = _rand!(rng, smp, M(undef, size(s)))
        end
    else
        for x in X
            rand!(rng, smp, x)
        end
    end
    return X
end

# multiple array-variates, must allocate array of arrays
rand(rng::AbstractRNG, s::Sampleable{ArrayLikeVariate{N}}, dims::Dims) where N =
    rand!(rng, s, Array{Array{N}{eltype(s)}}(undef, dims), true)
rand(rng::AbstractRNG, s::Sampleable{ArrayLikeVariate{N},Continuous}, dims::Dims) where N =
    rand!(rng, s, Array{Array{N}{float(eltype(s))}}(undef, dims), true)

# single array-variate, must allocate one array
rand(rng::AbstractRNG, s::Sampleable{ArrayLikeVariate{N}}) where N =
    _rand!(rng, s, Array{eltype(s)}(undef, size(s)))
rand(rng::AbstractRNG, s::Sampleable{ArrayLikeVariate{N},Continuous}) where N =
    _rand!(rng, s, Array{float(eltype(s))}(undef, size(s)))

# single matrix-variate with pre-allocated matrix
function rand!(rng::AbstractRNG, s::Sampleable{ArrayLikeVariate{N}},
               A::AbstractArray{<:Real,N}) where N
    @boundscheck size(A) == size(s) ||
        throw(DimensionMismatch("Output size inconsistent with matrix size."))
    return _rand!(rng, s, A)
end

# pdf & logpdf

_logpdf(d::MatrixDistribution, X::AbstractMatrix) = logkernel(d, X) + d.logc0

_pdf(d::MatrixDistribution, x::AbstractMatrix{T}) where {T<:Real} = exp(_logpdf(d, x))

"""
    logpdf(d::MatrixDistribution, AbstractMatrix)

Compute the logarithm of the probability density at the input matrix `x`.
"""
function logpdf(d::ArrayDistribution{<:Any,N}, x::AbstractArray{T,N}) where {T<:Real,N}
    size(x) == size(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _logpdf(d, x)
end

"""
    pdf(d::MatrixDistribution, x::AbstractArray)

Compute the probability density at the input matrix `x`.
"""
function pdf(d::ArrayDistribution{<:Any,N}, x::AbstractArray{T,N}) where {T<:Real,N}
    size(x) == size(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _pdf(d, x)
end

function _logpdf!(r::AbstractArray, d::ArrayDistribution{<:Any,N}, X::AbstractArray{<:AbstractArray{<:Real,N}}) where N
    r .= logpdf.(Ref(d), X)
    return r
end

function _pdf!(r::AbstractArray, d::ArrayDistribution{<:Any,N}, X::AbstractArray{<:AbstractArray{<:Real,N}}) where N
    r .= pdf.(Ref(d), X)
    return r
end

function logpdf!(r::AbstractArray, d::ArrayDistribution{<:Any,N}, X::AbstractArray{<:AbstractArray{<:Real,N}}) where N
    length(X) == length(r) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _logpdf!(r, d, X)
end

function pdf!(r::AbstractArray, d::ArrayDistribution{<:Any,N}, X::AbstractArray{<:AbstractArray{<:Real,N}}) where N
    length(X) == length(r) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _pdf!(r, d, X)
end

function logpdf(d::ArrayDistribution{<:Any,N}, X::AbstractArray{<:AbstractArray{<:Real,N}}) where N
    map(Base.Fix1(logpdf, d), X)
end

function pdf(d::ArrayDistribution{<:Any,N}, X::AbstractArray{<:AbstractArray{<:Real,N}}) where N
    map(Base.Fix1(pdf, d), X)
end

"""
    _logpdf(d::ArrayDistribution{S,N}, x::AbstractArray{<:Real,N})

Evaluate logarithm of pdf value for a given sample `x`. This function need not perform dimension checking.
"""
_logpdf(d::ArrayDistribution{S,N}, x::AbstractArray{<:Real,N}) where {S,N}

"""
    loglikelihood(d::ArrayDistribution, x::AbstractArray)

The log-likelihood of distribution `d` with respect to all samples contained in array `x`.

Here, `x` can be a matrix of size `size(d)`, a three-dimensional array with `size(d, 1)`
rows and `size(d, 2)` columns, or an array of matrices of size `size(d)`.
"""
loglikelihood(d::ArrayDistribution{<:Any,N}, X::AbstractArray{<:Real,N}) where N = logpdf(d, X)
function loglikelihood(d::MatrixDistribution, X::AbstractArray{<:Real,3})
    (size(X, 1), size(X, 2)) == size(d) || throw(DimensionMismatch("Inconsistent array dimensions."))
    return sum(i -> _logpdf(d, view(X, :, :, i)), axes(X, 3))
end
function loglikelihood(d::ArrayDistribution{<:Any,N}, X::AbstractArray{<:AbstractArray{<:Real,N}}) where N
    return sum(x -> logpdf(d, x), X)
end

#  for testing
is_univariate(d::ArrayDistribution{<:Any,N}) where N = prod(size(d)) == ntuple(_ -> 1, Val(N))
check_univariate(d::ArrayDistribution) = is_univariate(d) || throw(ArgumentError("not 1 x 1"))

##### Specific distributions #####

for fname in ["wishart.jl", "inversewishart.jl", "matrixnormal.jl",
              "arrayreshaped.jl", "matrixtdist.jl", "matrixbeta.jl", 
              "matrixfdist.jl", "lkj.jl"]
    include(joinpath("matrix", fname))
end
