"""
    ArrayReshaped(D, dims...)
```julia
* `D::MultivariateDistribution`:  base distribution
* `dims::Integer...`: dimensions (size) of the resulting distribution, e.g. `n, m` for
   a `Matrixvariate` distribution with `n` rows and `m` columns.
```
Reshapes a multivariate distribution into an array distribution with arrays of size
`dims` as variates resp. samples.
"""
struct ArrayReshaped{S<:ValueSupport,D<:MultivariateDistribution{S},N} <:
       Distribution{ArrayLikeVariate{N},S}
    d::D
    dims::NTuple{N,Int}
    function ArrayReshaped(
        d::D,
        dims::NTuple{N,Integer},
    ) where {
        D<:MultivariateDistribution{S},
        N
    } where {S<:ValueSupport}
        all(x -> x>0, dims) || throw(ArgumentError("dimensions must be positive"))
        prod(dims) == length(d) ||
        throw(ArgumentError("Dimensions $dims provided do not match source distribution of length $(length(d))"))
        return new{S,D,N}(d, dims)
    end
end

ArrayReshaped(d::MultivariateDistribution, dims::Integer...) = ArrayReshaped(d, dims)

const MatrixReshaped{S<:ValueSupport,D<:MultivariateDistribution{S}} = ArrayReshaped{S,D,2}

MatrixReshaped(D::MultivariateDistribution, n::Integer, m::Integer) = ArrayReshaped(D, n, m)
MatrixReshaped(D::MultivariateDistribution, n::Integer) = ArrayReshaped(D, n, n)

show(io::IO, d::ArrayReshaped) = show_multline(io, d, [(:dims, d.dims)])


#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

size(d::ArrayReshaped) = d.dims

length(d::ArrayReshaped) = prod(size(d))

rank(d::ArrayReshaped) = minimum(size(d))

function insupport(d::ArrayReshaped, X::AbstractArray) where N
    return isreal(X) && size(d) == size(X) && insupport(d.d, view(X, :))
end

mean(d::ArrayReshaped) = reshape(mean(d.d), size(d))
mode(d::ArrayReshaped) = reshape(mode(d.d), size(d))
cov(d::ArrayReshaped, ::Val{true} = Val(true)) = reshape(cov(d.d), prod(size(d)), prod(size(d)))
cov(d::ArrayReshaped, ::Val{false}) = reshape(cov(d), size(d)..., size(d)...)
var(d::ArrayReshaped) = reshape(var(d.d), size(d))

params(d::ArrayReshaped) = (d.d, d.dims...)

@inline partype(
    d::ArrayReshaped{S,<:MultivariateDistribution{S}},
) where {S<:Real} = S

_logpdf(d::ArrayReshaped{<:Any,<:Any,N}, X::AbstractArray{<:Any,N}) where N = logpdf(d.d, view(X, :))
_logpdf(d::MatrixReshaped, X::AbstractMatrix) = logpdf(d.d, view(X, :))

_pdf(d::ArrayReshaped{<:Any,<:Any,N}, X::AbstractArray{<:Any,N}) where N = pdf(d.d, view(X, :))
_pdf(d::MatrixReshaped, X::AbstractMatrix) = pdf(d.d, view(X, :))

function _rand!(rng::AbstractRNG, d::ArrayReshaped, Y::AbstractArray)
    rand!(rng, d.d, view(Y, :))
    return Y
end

vec(d::ArrayReshaped) = d.d
