"""
    LogBNormal(μ,σ,B)

The *LogB normal distribution* is the distribution of the base `B` exponential of a [`Normal`](@ref) variate: if ``X \\sim \\operatorname{Normal}(\\mu, \\sigma)`` then
``B^x \\sim \\operatorname{LogBNormal}(\\mu,\\sigma,B)``. The probability density function is
```math
f(x; \\mu, \\sigma) = \\frac{1}{x \\log(B) \\sqrt{2 \\pi \\sigma^2}}
\\exp \\left( - \\frac{(\\log(B,x) - \\mu)^2}{2 \\sigma^2} \\right),
\\quad x > 0
```
```julia
LogBNormal()          # LogB-normal distribution with zero log-mean, unit scale, and base ℯ
LogBNormal(μ)         # LogB-normal distribution with log-mean `μ`, unit scale, and base ℯ
LogBNormal(μ, σ)      # LogB-normal distribution with log-mean `μ`, scale `σ`, and base ℯ
LogBNormal(μ, σ, B)   # LogB-normal distribution with log-mean `μ`, scale `σ`, and base B

params(d)              # Get the parameters, i.e. (μ, σ, B)
```
External links

* [Log normal distribution on Wikipedia](http://en.wikipedia.org/wiki/Log-normal_distribution)

"""
struct LogBNormal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    B::T
    LogBNormal{T}(μ::T, σ::T, B::T) where {T} = new{T}(μ, σ, B)
end

function LogBNormal(μ::T, σ::T, B::T; check_args::Bool=true) where {T <: Real}
    @check_args LogBNormal (σ, σ ≥ zero(σ)) (B, B ≥ zero(B))
    return LogBNormal{T}(μ, σ, B)
end

LogBNormal(μ::Real, σ::Real, B::Real=ℯ; check_args::Bool=true) = LogBNormal(promote(μ, σ, B)...; check_args=check_args)
LogBNormal(μ::Real=0.0) = LogBNormal(promote(μ, one(μ), ℯ)...; check_args=false)

LogBNormal(μ::Integer, σ::Integer, B::Integer; check_args::Bool=true) = LogBNormal(float(μ), float(σ), float(B); check_args=check_args)
LogBNormal(μ::Integer, σ::Integer; check_args::Bool=true) = LogBNormal(float(μ), float(σ), float(ℯ); check_args=check_args)
LogBNormal(μ::Integer)= LogBNormal(float(μ), 1.0, float(ℯ); check_args=false)

@distr_support LogBNormal 0.0 Inf

#### Conversions
convert(::Type{LogBNormal{T}}, μ::S, σ::S, B::S) where {T <: Real, S <: Real} = LogBNormal(T(μ), T(σ), T(B))
Base.convert(::Type{LogBNormal{T}}, d::LogBNormal) where {T<:Real} = LogBNormal{T}(T(d.μ), T(d.σ), T(d.B))
Base.convert(::Type{LogBNormal{T}}, d::LogBNormal{T}) where {T<:Real} = d
# Change of base rule
Base.convert(d::LogBNormal{T}, B::Real) where {T<:Real} = LogBNormal{T}(T(d.μ/log(B)*log(d.B)), T(d.σ/log(B)*log(d.B)), T(B))

#### Parameters

params(d::LogBNormal) = (d.μ, d.σ, d.B)
partype(::LogBNormal{T}) where {T} = T

#### Statistics

mean(d::LogBNormal) = ((μ, σ, B) = params(d); B^(μ + σ^2*log(B)/2))
median(d::LogBNormal) = d.B^d.μ
mode(d::LogBNormal) = ((μ, σ, B) = params(d); B^(μ - σ^2*log(B)))
# testmean(d::LogBNormal) = quadgk(x->d.B^2x * log(d.B) * pdf(d,d.B^x), min(0.0,d.μ-5*d.σ),d.μ+5*d.σ)[1]

function var(d::LogBNormal)
    σ2 = d.σ^2
    B = d.B
    σ2 *= log(one(σ2)*B)^2
    e = exp(σ2)
    μ = d.μ
    B^2μ * e * (e-1)
end
# testvar(d::LogBNormal) = quadgk(x->d.B^x * (d.B^x-mean(d))^2 * log(d.B) * pdf(d,d.B^x), min(0.0,d.μ-5*d.σ),d.μ+5*d.σ)[1]

function skewness(d::LogBNormal)
    σ2 = d.σ^2
    B = d.B
    σ2 *= log(one(σ2)*B)^2
    e = exp(σ2)
    (e + 2) * sqrt(e - 1)
end
# testskew(d::LogBNormal) = quadgk(x->d.B^x * (d.B^x-mean(d))^3 * log(d.B) * pdf(d,d.B^x), min(0.0,d.μ-5*d.σ),d.μ+5*d.σ)[1] / quadgk(x->d.B^x * (d.B^x-mean(d))^2 * log(d.B) * pdf(d,d.B^x), min(0.0,d.μ-5*d.σ),d.μ+5*d.σ)[1]^(3/2)

function kurtosis(d::LogBNormal)
    σ2 = d.σ^2
    B = d.B
    σ2 *= log(one(σ2)*B)^2
    e = exp(σ2)
    e2 = e * e
    e3 = e2 * e
    e4 = e3 * e
    e4 + 2*e3 + 3*e2 - 3
end
# testkurtosis(d::LogBNormal) = quadgk(x->d.B^x * (d.B^x-mean(d))^4 * log(d.B) * pdf(d,d.B^x), min(0.0,d.μ-5*d.σ),d.μ+5*d.σ)[1] / var(d)^2

function entropy(d::LogBNormal)
    (μ, σ, B) = params(d)
    lnB=log(B)
    # (1 + μ * log(100) + log(twoπ) + 2*log(σ*log(10)))/2
    (1 + μ * 2*lnB + log(twoπ) + 2*log(σ*lnB))/2
end
# testentropy(d::LogBNormal) = quadgk(x->d.B^x * log(d.B)* -logpdf(d,d.B^x) * pdf(d,d.B^x), min(0.0,d.μ-5*d.σ),d.μ+5*d.σ)[1] 

#### Evaluation

function pdf(d::LogBNormal, x::Real)
    (μ, σ, B) = params(d)
    if x ≤ zero(x)
        logbx = log(B,zero(x))
        x = one(x)
    else
        logbx = log(B,x)
    end
    return pdf(Normal(d.μ, d.σ), logbx) / x / log(one(x)*B)
end
# testpdf(d::LogBNormal) = quadgk(x->d.B^x * log(d.B)* pdf(d,d.B^x), min(0.0,d.μ-5*d.σ),d.μ+5*d.σ)[1]

function logpdf(d::LogBNormal, x::Real)
    (μ, σ, B) = params(d)
    if x ≤ zero(x)
        logbx = log(B,zero(x))
        b = zero(logbx)
    else
        logbx = log(B,x)
        b = log(x)
    end
    return logpdf(Normal(d.μ, d.σ), logbx) - b - log(log(one(x)*B))
end
# testlogpdf(d::LogBNormal,x::Real) = (isapprox(log(pdf(d,x)),logpdf(d,x);atol=1e-12))

function cdf(d::LogBNormal, x::Real)
    (μ, σ, B) = params(d)
    logbx = x ≤ zero(x) ? log(B,zero(x)) : log(B,x)
    return cdf(Normal(μ, σ), logbx)
end

function ccdf(d::LogBNormal, x::Real)
    (μ, σ, B) = params(d)
    logbx = x ≤ zero(x) ? log(B,zero(x)) : log(B,x)
    return ccdf(Normal(μ, σ), logbx)
end

function logcdf(d::LogBNormal, x::Real)
    (μ, σ, B) = params(d)
    logbx = x ≤ zero(x) ? log(B,zero(x)) : log(B,x)
    return logcdf(Normal(μ, σ), logbx)
end

function logccdf(d::LogBNormal, x::Real)
    (μ, σ, B) = params(d)
    logbx = x ≤ zero(x) ? log(B,zero(x)) : log(B,x)
    return logccdf(Normal(μ, σ), logbx)
end

quantile(d::LogBNormal, q::Real) = d.B^(quantile(Normal(d.μ,d.σ), q))
cquantile(d::LogBNormal, q::Real) = d.B^(cquantile(Normal(d.μ,d.σ), q))
invlogcdf(d::LogBNormal, lq::Real) = d.B^(invlogcdf(Normal(d.μ,d.σ), lq))
invlogccdf(d::LogBNormal, lq::Real) = d.B^(invlogccdf(Normal(d.μ,d.σ), lq))

function gradlogpdf(d::LogBNormal, x::Real)
    outofsupport = x ≤ zero(x)
    y = outofsupport ? one(x) : x
    (μ, σ, B) = params(d)
    lnB = log(one(μ)*B)
    z = ( lnB * (μ - σ^2 * lnB) - log(y) ) / (y * σ^2 * lnB^2)
    return outofsupport ? zero(z) : z
end

#### Sampling

rand(rng::AbstractRNG, d::LogBNormal) = d.B^(randn(rng) * d.σ + d.μ)

## Fitting

"""
    fit_mle(::Type{<:LogBNormal}, x::AbstractArray{<:Real}, [w::AbstractArray{<:Real}], B::Real=ℯ)

Compute the maximum likelihood estimate for a lognormal distribution with base `B`. 
"""
function fit_mle(::Type{<:LogBNormal}, x::AbstractArray{T}, B::Real=ℯ) where T<:Real
    lx = log.(B,x)
    μ, σ = mean_and_std(lx)
    LogBNormal(μ, σ, B)
end

function fit_mle(::Type{<:LogBNormal}, x::AbstractArray{T}, w::AbstractArray{S}, B::Real=ℯ) where {T<:Real,S<:Real}
    @assert size(x) == size(w)
    lx = log.(B,x)
    μ = sum( lx .* w ) / sum(w)
    σ = sqrt( sum( (lx .- μ ).^2 .* w) / sum(w) )
    LogBNormal(μ, σ, B)
end
