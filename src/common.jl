## sample space/domain

"""
`F <: VariateForm` specifies the form or shape of the variate or a sample.
"""
abstract type VariateForm end

"""
`F <: ArrayLikeVariate{N}` specifies the number of axes of a variate or
a sample with an array-like shape, e.g. univariate (scalar, `N == 0`),
multivariate (vector, `N == 1`) or matrix-variate (matrix, `N == 2`).
"""
abstract type ArrayLikeVariate{N} <: VariateForm end

const Univariate    = ArrayLikeVariate{0}
const Multivariate  = ArrayLikeVariate{1}
const Matrixvariate = ArrayLikeVariate{2}

"""
`F <: CholeskyVariate` specifies that the variate or a sample is of type
`LinearAlgebra.Cholesky`.
"""
abstract type CholeskyVariate <: VariateForm end

"""
`S <: ValueSupport` specifies the support of sample elements,
either discrete or continuous.
"""
abstract type ValueSupport end
struct Discrete   <: ValueSupport end
struct Continuous <: ValueSupport end

## Sampleable

"""
    Sampleable{F<:VariateForm,S<:ValueSupport}

`Sampleable` is any type able to produce random values.
Parametrized by a `VariateForm` defining the dimension of samples
and a `ValueSupport` defining the domain of possibly sampled values.
Any `Sampleable` implements the `Base.rand` method.
"""
abstract type Sampleable{F<:VariateForm,S<:ValueSupport} end

"""
    length(s::Sampleable)

The length of each sample. Always returns `1` when `s` is univariate.
"""
Base.length(s::Sampleable) = prod(size(s))
Base.length(::Sampleable{Univariate}) = 1
Base.length(s::Sampleable{Multivariate}) = throw(MethodError(length, (s,)))

"""
    size(s::Sampleable)

The size (i.e. shape) of each sample. Always returns `()` when `s` is univariate, and
`(length(s),)` when `s` is multivariate.
"""
Base.size(s::Sampleable)
Base.size(s::Sampleable{Univariate}) = ()
Base.size(s::Sampleable{Multivariate}) = (length(s),)

"""
    eltype(::Type{Sampleable})

The default element type of a sample. This is the type of elements of the samples generated
by the `rand` method. However, one can provide an array of different element types to
store the samples using `rand!`.
"""
Base.eltype(::Type{<:Sampleable{F,Discrete}}) where {F} = Int
Base.eltype(::Type{<:Sampleable{F,Continuous}}) where {F} = Float64

"""
    nsamples(s::Sampleable)

The number of values contained in one sample of `s`. Multiple samples are often organized
into an array, depending on the variate form.
"""
nsamples(t::Type{Sampleable}, x::Any)
nsamples(::Type{D}, x::Number) where {D<:Sampleable{Univariate}} = 1
nsamples(::Type{D}, x::AbstractArray) where {D<:Sampleable{Univariate}} = length(x)
nsamples(::Type{D}, x::AbstractVector) where {D<:Sampleable{Multivariate}} = 1
nsamples(::Type{D}, x::AbstractMatrix) where {D<:Sampleable{Multivariate}} = size(x, 2)
nsamples(::Type{D}, x::Number) where {D<:Sampleable{Matrixvariate}} = 1
nsamples(::Type{D}, x::Array{Matrix{T}}) where {D<:Sampleable{Matrixvariate},T<:Number} = length(x)

for func in (:(==), :isequal, :isapprox)
    @eval function Base.$func(s1::A, s2::B; kwargs...) where {A<:Sampleable, B<:Sampleable}
        nameof(A) === nameof(B) || return false
        fields = fieldnames(A)
        fields === fieldnames(B) || return false

        for f in fields
            isdefined(s1, f) && isdefined(s2, f) || return false
            $func(getfield(s1, f), getfield(s2, f); kwargs...) || return false
        end

        return true
    end
end

function Base.hash(s::S, h::UInt) where S <: Sampleable
    hashed = hash(Sampleable, h)
    hashed = hash(nameof(S), hashed)

    for f in fieldnames(S)
        hashed = hash(getfield(s, f), hashed)
    end

    return hashed
end

"""
    Distribution{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S}

`Distribution` is a `Sampleable` generating random values from a probability
distribution. Distributions define a Probability Distribution Function (PDF)
to implement with `pdf` and a Cumulated Distribution Function (CDF) to implement
with `cdf`.
"""
abstract type Distribution{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S} end

const UnivariateDistribution{S<:ValueSupport}   = Distribution{Univariate,S}
const MultivariateDistribution{S<:ValueSupport} = Distribution{Multivariate,S}
const MatrixDistribution{S<:ValueSupport}       = Distribution{Matrixvariate,S}
const NonMatrixDistribution = Union{UnivariateDistribution, MultivariateDistribution}

const DiscreteDistribution{F<:VariateForm}   = Distribution{F,Discrete}
const ContinuousDistribution{F<:VariateForm} = Distribution{F,Continuous}

const DiscreteUnivariateDistribution     = Distribution{Univariate,    Discrete}
const ContinuousUnivariateDistribution   = Distribution{Univariate,    Continuous}
const DiscreteMultivariateDistribution   = Distribution{Multivariate,  Discrete}
const ContinuousMultivariateDistribution = Distribution{Multivariate,  Continuous}
const DiscreteMatrixDistribution         = Distribution{Matrixvariate, Discrete}
const ContinuousMatrixDistribution       = Distribution{Matrixvariate, Continuous}

variate_form(::Type{<:Distribution{VF}}) where {VF} = VF

value_support(::Type{<:Distribution{VF,VS}}) where {VF,VS} = VS

# allow broadcasting over distribution objects
# to be decided: how to handle multivariate/matrixvariate distributions?
Broadcast.broadcastable(d::UnivariateDistribution) = Ref(d)

"""
    minimum(d::Distribution)

Return the minimum of the support of `d`.
"""
minimum(d::Distribution)

"""
    maximum(d::Distribution)

Return the maximum of the support of `d`.
"""
maximum(d::Distribution)

"""
    extrema(d::Distribution)

Return the minimum and maximum of the support of `d` as a 2-tuple.
"""
Base.extrema(d::Distribution) = minimum(d), maximum(d)

"""
    pdf(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,N}) where {N}

Evaluate the probability density function of `d` at `x`.

This function checks if the size of `x` is compatible with distribution `d`. This check can
be disabled by using `@inbounds`.

# Implementation

Instead of `pdf` one should implement `_pdf(d, x)` which does not have to check the size of
`x`. However, due to the fallback `_pdf(d, x) = exp(_logpdf(d, x))` usually it is sufficient
to implement `_logpdf`.

See also: [`logpdf`](@ref).
"""
@inline function pdf(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,N}
) where {N}
    @boundscheck begin
        size(x) == size(d) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
    end
    return _pdf(d, x)
end

function _pdf(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,N}) where {N}
    return exp(@inbounds logpdf(d, x))
end

"""
    logpdf(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,N}) where {N}

Evaluate the probability density function of `d` at `x`.

This function checks if the size of `x` is compatible with distribution `d`. This check can
be disabled by using `@inbounds`.

# Implementation

Instead of `logpdf` one should implement `_logpdf(d, x)` which does not have to check the
size of `x`.

See also: [`pdf`](@ref).
"""
@inline function logpdf(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,N}
) where {N}
    @boundscheck begin
        size(x) == size(d) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
    end
    return _logpdf(d, x)
end

# `_logpdf` should be implemented and has no default definition
# _logpdf(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,N}) where {N}

# TODO: deprecate?
"""
    pdf(d::Distribution{<:ArrayLikeVariate}, x)

Evaluate the probability density function of `d` at every element in a collection `x`.

This function checks for every element of `x` if its size is compatible with distribution
`d`. This check can be disabled by using `@inbounds`.
"""
Base.@propagate_inbounds function pdf(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:AbstractArray{<:Real,N}},
) where {N}
    return map(Base.Fix1(pdf, d), x)
end

@inline function pdf(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,M},
) where {N,M}
    @boundscheck begin
        M > N ||
            throw(DimensionMismatch(
                "number of dimensions of `x` ($M) must be greater than number of dimensions of `d` ($N)"
            ))
        ntuple(i -> size(x, i), Val(N)) == size(d) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
    end
    return @inbounds map(Base.Fix1(pdf, d), eachvariate(x, variate_form(typeof(d))))
end

"""
    logpdf(d::Distribution{<:ArrayLikeVariate}, x)

Evaluate the logarithm of the probability density function of `d` at every element in a
collection `x`.

This function checks for every element of `x` if its size is compatible with distribution
`d`. This check can be disabled by using `@inbounds`.
"""
Base.@propagate_inbounds function logpdf(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:AbstractArray{<:Real,N}},
) where {N}
    return map(Base.Fix1(logpdf, d), x)
end

@inline function logpdf(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,M},
) where {N,M}
    @boundscheck begin
        M > N ||
            throw(DimensionMismatch(
                "number of dimensions of `x` ($M) must be greater than number of dimensions of `d` ($N)"
            ))
        ntuple(i -> size(x, i), Val(N)) == size(d) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
    end
    return @inbounds map(Base.Fix1(logpdf, d), eachvariate(x, variate_form(typeof(d))))
end

"""
    pdf!(
        out::AbstractArray{<:Real},
        d::Distribution{<:ArrayLikeVariate},
        x,
    )

Compute the
    x::AbstractArray{<:AbstractArray{<:Real,N}})
"""
Base.@propagate_inbounds function pdf!(
    out::AbstractArray{<:Real,M},
    d::Distribution{ArrayLikeVariate{N}},
    x::AbstractArray{<:AbstractArray{<:Real,N},M}
) where {N,M}
    return map!(Base.Fix1(pdf, d), out, x)
end

Base.@propagate_inbounds function logpdf!(
    out::AbstractArray{<:Real,M},
    d::Distribution{ArrayLikeVariate{N}},
    x::AbstractArray{<:AbstractArray{<:Real,N},M}
) where {N,M}
    return map!(Base.Fix1(logpdf, d), out, x)
end

@inline function pdf!(
    out::AbstractArray{<:Real,M},
    d::Distribution{ArrayLikeVariate{N}},
    x::AbstractArray{<:Real},
) where {N,M}
    @boundscheck begin
        ndims(x) == N + M || throw(
            DimensionMismatch(
                "number of dimensions of `x` ($(ndims(x))) must be equal to the sum of " *
                "the dimensions of `d` ($N) and output `out` ($M)"
            )
        )
        ntuple(i -> size(x, i), Val(N)) == size(d) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
        ntuple(i -> size(x, i + N), Val(M)) == size(out) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
    end
    return _pdf!(out, d, X)
end

function _pdf!(
    out::AbstractArray{<:Real},
    d::Distribution{<:ArrayLikeVariate},
    x::AbstractArray{<:Real},
)
    @inbounds logpdf!(out, d, x)
    map!(exp, out, out)
    return out
end

@inline function logpdf!(
    out::AbstractArray{<:Real,M},
    d::Distribution{ArrayLikeVariate{N}},
    x::AbstractArray{<:Real},
) where {N,M}
    @boundscheck begin
        ndims(x) == N + M || throw(
            DimensionMismatch(
                "number of dimensions of `x` ($(ndims(x))) must be equal to the sum of " *
                "the dimensions of `d` ($N) and output `out` ($M)"
            )
        )
        ntuple(i -> size(x, i), Val(N)) == size(d) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
        ntuple(i -> size(x, i + N), Val(M)) == size(out) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
    end
    return _logpdf!(out, d, X)
end

# default definition
function _logpdf!(
    out::AbstractArray{<:Real},
    d::Distribution{<:ArrayLikeVariate},
    x::AbstractArray{<:Real},
)
    @inbounds map!(Base.Fix1(logpdf, d), out, eachvariate(x, variate_form(typeof(d))))
    return out
end

"""
    loglikelihood(d::Distribution{ArrayLikeVariate{N}}, x) where {N}

The log-likelihood of distribution `d` with respect to all variate(s) contained in `x`.

Here, `x` can be any output of `rand(d, dims...)` and `rand!(d, x)`. For instance, `x` can
be
- an array of dimension `N` with `size(x) == size(d)`,
- an array of dimension `N + 1` with `size(x)[1:N] == size(d)`, or
- an array of arrays `xi` of dimension `N` with `size(xi) == size(d)`.
"""
Base.@propagate_inbounds function loglikelihood(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,N},
) where {N}
    return logpdf(d, x)
end
@inline function loglikelihood(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,M},
) where {N,M}
    @boundscheck begin
        M > N ||
            throw(DimensionMismatch(
                "number of dimensions of `x` ($M) must be greater than number of dimensions of `d` ($N)"
            ))
        ntuple(i -> size(x, i), Val(N)) == size(d) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
    end
    # we use pairwise summation (https://github.com/JuliaLang/julia/pull/31020)
    # to compute `sum(logpdf.((d,), eachvariate(x, V)))`
    @inbounds broadcasted = Broadcast.broadcasted(
        logpdf, (d,), eachvariate(x, ArrayLikeVariate{N}),
    )
    return sum(Broadcast.instantiate(broadcasted))
end
Base.@propagate_inbounds function loglikelihood(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:AbstractArray{<:Real,N}},
) where {N}
    # we use pairwise summation (https://github.com/JuliaLang/julia/pull/31020)
    # to compute `sum(logpdf.((d,), x))`
    broadcasted = Broadcast.broadcasted(logpdf, (d,), x)
    return sum(Broadcast.instantiate(broadcasted))
end

## TODO: the following types need to be improved
abstract type SufficientStats end
abstract type IncompleteDistribution end

const DistributionType{D<:Distribution} = Type{D}
const IncompleteFormulation = Union{DistributionType,IncompleteDistribution}

"""
    succprob(d::DiscreteUnivariateDistribution)

Get the probability of success.
"""
succprob(d::DiscreteUnivariateDistribution)

"""
    failprob(d::DiscreteUnivariateDistribution)

Get the probability of failure.
"""
failprob(d::DiscreteUnivariateDistribution)

# Temporary fix to handle RFunctions dependencies
"""
    @rand_rdist(::Distribution)

Mark a `Distribution` subtype as requiring RFunction calls. Since these calls
cannot accept an arbitrary random number generator as an input, this macro
creates new `rand(::Distribution, n::Int)` and
`rand!(::Distribution, X::AbstractArray)` functions that call the relevant
RFunction. Calls using another random number generator still work, but rely on
a quantile function to operate.
"""
macro rand_rdist(D)
    esc(quote
        function rand(d::$D, n::Int)
            [rand(d) for i in Base.OneTo(n)]
        end
        function rand!(d::$D, X::AbstractArray)
            for i in eachindex(X)
                X[i] = rand(d)
            end
            return X
        end
    end)
end
