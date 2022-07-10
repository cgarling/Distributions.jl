using Distributions
using ForwardDiff
using Test

isnan_type(::Type{T}, v) where {T} = isnan(v) && v isa T

@testset "LogBNormal" begin
    @test isa(convert(LogBNormal{Float64}, Float16(0), Float16(1), Float16(10)),
              LogBNormal{Float64})
    d = LogBNormal(0, 1, 10)
    @test convert(LogBNormal{Float64}, d) === d
    @test convert(LogBNormal{Float32}, d) isa LogBNormal{Float32}
    # test change of base
    let d = LogBNormal(1.0,0.2,3.0), c = convert(d,10.0)
        @test mean(d) ≈ mean(c) rtol = 1e-12
        @test median(d) ≈ median(c) rtol = 1e-12
        @test mode(d) ≈ mode(c) rtol = 1e-12
        @test var(d) ≈ var(c) rtol = 1e-12
        @test skewness(d) ≈ skewness(c) rtol = 1e-12
        @test kurtosis(d) ≈ kurtosis(c) rtol = 1e-12
        @test pdf(d,1.0) ≈ pdf(c,1.0) rtol = 1e-12
        @test logpdf(d,1.0) ≈ logpdf(c,1.0) rtol = 1e-12
        @test cdf(d,1.0) ≈ cdf(c,1.0) rtol = 1e-12
        @test ccdf(d,1.0) ≈ ccdf(c,1.0) rtol = 1e-12
        @test logccdf(d,1.0) ≈ logccdf(c,1.0) rtol = 1e-12
        @test quantile(d,1.0) ≈ quantile(c,1.0) rtol = 1e-12
        @test cquantile(d,1.0) ≈ cquantile(c,1.0) rtol = 1e-12
        @test invlogcdf(d,logcdf(d,1.0)) ≈ invlogcdf(c,logcdf(d,1.0)) rtol = 1e-12
        @test invlogccdf(d,logccdf(d,1.0)) ≈ invlogccdf(c,logccdf(d,1.0)) rtol = 1e-12
        @test gradlogpdf(d,1.0) ≈ gradlogpdf(c,1.0) rtol = 1e-12
    end

    @test logpdf(LogBNormal(0, 0, ℯ), 1) === Inf
    @test logpdf(LogBNormal(0, 0, 10), 1) === Inf
    @test logpdf(LogBNormal(), Inf) === -Inf
    @test iszero(logcdf(LogBNormal(0, 0, ℯ), 1))
    @test iszero(logcdf(LogBNormal(0, 0, 10), 1))
    @test iszero(logcdf(LogBNormal(), Inf))
    @test logdiffcdf(LogBNormal(), Float32(exp(3)), Float32(exp(3))) === -Inf
    @test logdiffcdf(LogBNormal(), Float32(exp(5)), Float32(exp(3))) ≈ -6.607938594596893 rtol=1e-8
    @test logdiffcdf(LogBNormal(), Float32(exp(5)), Float64(exp(3))) ≈ -6.60793859457367 rtol=1e-12
    @test logdiffcdf(LogBNormal(), Float64(exp(5)), Float64(exp(3))) ≈ -6.607938594596893 rtol=1e-12
    let d = LogBNormal(Float64(0), Float64(1), Float64(ℯ)), x = Float64(exp(-60)), y = Float64(exp(-60.001))
        float_res = logdiffcdf(d, x, y)
        big_x = BigFloat(x; precision=100)
        big_y = BigFloat(y; precision=100)
        big_float_res = log(cdf(d, big_x) - cdf(d, big_y))
        @test float_res ≈ big_float_res
    end

    @test logccdf(LogBNormal(0, 0, ℯ), 1) === -Inf
    @test logccdf(LogBNormal(0, 0, 10), 1) === -Inf
    @test iszero(logccdf(LogBNormal(eps(), 0, ℯ), 1))
    @test iszero(logccdf(LogBNormal(eps(), 0, 10), 1))

    @test iszero(quantile(LogBNormal(), 0))
    @test iszero(quantile(LogBNormal(0,1,10), 0))
    @test isone(quantile(LogBNormal(), 0.5))
    @test isone(quantile(LogBNormal(0,1,10), 0.5))
    @test quantile(LogBNormal(), 1) === Inf
    @test quantile(LogBNormal(0,1,10), 1) === Inf

    @test iszero(quantile(LogBNormal(0, 0, ℯ), 0))
    @test iszero(quantile(LogBNormal(0, 0, 10), 0))
    @test isone(quantile(LogBNormal(0, 0, ℯ), 0.75))
    @test isone(quantile(LogBNormal(0, 0, 10), 0.75))
    @test quantile(LogBNormal(0, 0, ℯ), 1) === Inf
    @test quantile(LogBNormal(0, 0, 10), 1) === Inf

    @test iszero(quantile(LogBNormal(0.25, 0, ℯ), 0))
    @test iszero(quantile(LogBNormal(0.25, 0, 10), 0))
    @test quantile(LogBNormal(0.25, 0, ℯ), 0.95) == exp(0.25)
    @test quantile(LogBNormal(0.25, 0, 10), 0.95) == exp10(0.25)
    @test quantile(LogBNormal(0.25, 0, 3), 0.95) == 3^(0.25)
    @test quantile(LogBNormal(0.25, 0, ℯ), 1) === Inf
    @test quantile(LogBNormal(0.25, 0, 10), 1) === Inf

    @test cquantile(LogBNormal(), 0) === Inf
    @test cquantile(LogBNormal(0,1,10), 0) === Inf
    @test isone(cquantile(LogBNormal(), 0.5))
    @test isone(cquantile(LogBNormal(0,1,10), 0.5))
    @test iszero(cquantile(LogBNormal(), 1))
    @test iszero(cquantile(LogBNormal(0,1,10), 1))

    @test cquantile(LogBNormal(0, 0, ℯ), 0) === Inf
    @test cquantile(LogBNormal(0, 0, 10), 0) === Inf
    @test isone(cquantile(LogBNormal(0, 0, ℯ), 0.75))
    @test isone(cquantile(LogBNormal(0, 0, 10), 0.75))
    @test iszero(cquantile(LogBNormal(0, 0, ℯ), 1))
    @test iszero(cquantile(LogBNormal(0, 0, 10), 1))

    @test cquantile(LogBNormal(0.25, 0, ℯ), 0) === Inf
    @test cquantile(LogBNormal(0.25, 0, 10), 0) === Inf
    @test cquantile(LogBNormal(0.25, 0, ℯ), 0.95) == exp(0.25)
    @test cquantile(LogBNormal(0.25, 0, 10), 0.95) == exp10(0.25)
    @test iszero(cquantile(LogBNormal(0.25, 0, ℯ), 1))
    @test iszero(cquantile(LogBNormal(0.25, 0, 10), 1))

    @test iszero(invlogcdf(LogBNormal(), -Inf))
    @test iszero(invlogcdf(LogBNormal(0,1,10), -Inf))
    @test isnan_type(Float64, invlogcdf(LogBNormal(), NaN))
    @test isnan_type(Float64, invlogcdf(LogBNormal(0,1,10), NaN))

    @test invlogccdf(LogBNormal(), -Inf) === Inf
    @test invlogccdf(LogBNormal(0,1,10), -Inf) === Inf
    @test isnan_type(Float64, invlogccdf(LogBNormal(), NaN))
    @test isnan_type(Float64, invlogccdf(LogBNormal(0,1,10), NaN))

    # test for #996 being fixed
    let d = LogBNormal(0, 1, ℯ), x = exp(1), ∂x = exp(2)
        @inferred cdf(d, ForwardDiff.Dual(x, ∂x)) ≈ ForwardDiff.Dual(cdf(d, x), ∂x * pdf(d, x))
    end
end


@testset "LogBNormal type inference" begin
    # pdf
    @test @inferred(pdf(LogBNormal(0.0, 0.0), 1.0)) === Inf
    @test @inferred(pdf(LogBNormal(0.0, 0.0), 0.5)) === 0.0
    @test @inferred(pdf(LogBNormal(0.0, 0.0), 0.0)) === 0.0
    @test @inferred(pdf(LogBNormal(0.0, 0.0), -0.5)) === 0.0

    @test @inferred(pdf(LogBNormal(0.0, 0.0), 1.0f0)) === Inf
    @test @inferred(pdf(LogBNormal(0.0f0, 0.0f0), 1.0)) === Inf
    @test @inferred(pdf(LogBNormal(0.0f0, 0.0f0, 10.0f0), 1.0)) === Inf
    @test @inferred(pdf(LogBNormal(0.0f0, 0.0f0), 1.0f0)) === Inf32

    @test isnan_type(Float64, @inferred(pdf(LogBNormal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(pdf(LogBNormal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(pdf(LogBNormal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(pdf(LogBNormal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(pdf(LogBNormal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(pdf(LogBNormal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(pdf(LogBNormal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(pdf(LogBNormal(0 // 1, 0 // 1), 1 // 1)) === Inf
    @test @inferred(pdf(LogBNormal(0 // 1, 0 // 1), 0 // 1)) === 0.0
    @test @inferred(pdf(LogBNormal(0 // 1, 0 // 1), -1 // 1)) === 0.0
    @test isnan_type(Float64, @inferred(pdf(LogBNormal(0 // 1, 0 // 1), NaN)))

    @test @inferred(pdf(LogBNormal(0.0, 0.0), BigInt(1))) == big(Inf)
    @test @inferred(pdf(LogBNormal(0.0, 0.0), BigInt(0))) == big(0.0)
    @test @inferred(pdf(LogBNormal(0.0, 0.0), BigInt(-1))) == big(0.0)
    @test @inferred(pdf(LogBNormal(0.0, 0.0), BigFloat(1))) == big(Inf)
    @test @inferred(pdf(LogBNormal(0.0, 0.0), BigFloat(0))) == big(0.0)
    @test @inferred(pdf(LogBNormal(0.0, 0.0), BigFloat(-1))) == big(0.0)
    @test isnan_type(BigFloat, @inferred(pdf(LogBNormal(0.0, 0.0), BigFloat(NaN))))

    # logpdf
    @test @inferred(logpdf(LogBNormal(0.0, 0.0), 1.0)) === Inf
    @test @inferred(logpdf(LogBNormal(0.0, 0.0), 0.5)) === -Inf
    @test @inferred(logpdf(LogBNormal(0.0, 0.0), 0.0)) === -Inf
    @test @inferred(logpdf(LogBNormal(0.0, 0.0), -0.5)) === -Inf

    @test @inferred(logpdf(LogBNormal(0.0, 0.0), 1.0f0)) === Inf
    @test @inferred(logpdf(LogBNormal(0.0f0, 0.0f0), 1.0)) === Inf
    @test @inferred(logpdf(LogBNormal(0.0f0, 0.0f0), 1.0f0)) === Inf32

    @test isnan_type(Float64, @inferred(logpdf(LogBNormal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(logpdf(LogBNormal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(logpdf(LogBNormal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(logpdf(LogBNormal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(logpdf(LogBNormal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(logpdf(LogBNormal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logpdf(LogBNormal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(logpdf(LogBNormal(0 // 1, 0 // 1), 1 // 1)) === Inf
    @test @inferred(logpdf(LogBNormal(0 // 1, 0 // 1), 0 // 1)) === -Inf
    @test @inferred(logpdf(LogBNormal(0 // 1, 0 // 1), -1 // 1)) === -Inf
    @test isnan_type(Float64, @inferred(logpdf(LogBNormal(0 // 1, 0 // 1), NaN)))

    @test @inferred(logpdf(LogBNormal(0.0, 0.0), BigInt(1))) == big(Inf)
    @test @inferred(logpdf(LogBNormal(0.0, 0.0), BigInt(0))) == big(-Inf)
    @test @inferred(logpdf(LogBNormal(0.0, 0.0), BigInt(-1))) == big(-Inf)
    @test @inferred(logpdf(LogBNormal(0.0, 0.0), BigFloat(1))) == big(Inf)
    @test @inferred(logpdf(LogBNormal(0.0, 0.0), BigFloat(0))) == big(-Inf)
    @test @inferred(logpdf(LogBNormal(0.0, 0.0), BigFloat(-1))) == big(-Inf)
    @test isnan_type(BigFloat, @inferred(logpdf(LogBNormal(0.0, 0.0), BigFloat(NaN))))

    # cdf
    @test @inferred(cdf(LogBNormal(0.0, 0.0), 1.0)) === 1.0
    @test @inferred(cdf(LogBNormal(0.0, 0.0), 0.5)) === 0.0
    @test @inferred(cdf(LogBNormal(0.0, 0.0), 0.0)) === 0.0
    @test @inferred(cdf(LogBNormal(0.0, 0.0), -0.5)) === 0.0

    @test @inferred(cdf(LogBNormal(0.0, 0.0), 1.0f0)) === 1.0
    @test @inferred(cdf(LogBNormal(0.0f0, 0.0f0), 1.0)) === 1.0
    @test @inferred(cdf(LogBNormal(0.0f0, 0.0f0), 1.0f0)) === 1.0f0

    @test isnan_type(Float64, @inferred(cdf(LogBNormal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(cdf(LogBNormal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(cdf(LogBNormal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(cdf(LogBNormal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(cdf(LogBNormal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(cdf(LogBNormal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(cdf(LogBNormal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(cdf(LogBNormal(0 // 1, 0 // 1), 1 // 1)) === 1.0
    @test @inferred(cdf(LogBNormal(0 // 1, 0 // 1), 0 // 1)) === 0.0
    @test @inferred(cdf(LogBNormal(0 // 1, 0 // 1), -1 // 1)) === 0.0
    @test isnan_type(Float64, @inferred(cdf(LogBNormal(0 // 1, 0 // 1), NaN)))

    @test @inferred(cdf(LogBNormal(0.0, 0.0), BigInt(1))) == big(1.0)
    @test @inferred(cdf(LogBNormal(0.0, 0.0), BigInt(0))) == big(0.0)
    @test @inferred(cdf(LogBNormal(0.0, 0.0), BigInt(-1))) == big(0.0)
    @test @inferred(cdf(LogBNormal(0.0, 0.0), BigFloat(1))) == big(1.0)
    @test @inferred(cdf(LogBNormal(0.0, 0.0), BigFloat(0))) == big(0.0)
    @test @inferred(cdf(LogBNormal(0.0, 0.0), BigFloat(-1))) == big(0.0)
    @test isnan_type(BigFloat, @inferred(cdf(LogBNormal(0.0, 0.0), BigFloat(NaN))))

    # logcdf
    @test @inferred(logcdf(LogBNormal(0.0, 0.0), 1.0)) === -0.0
    @test @inferred(logcdf(LogBNormal(0.0, 0.0), 0.5)) === -Inf
    @test @inferred(logcdf(LogBNormal(0.0, 0.0), 0.0)) === -Inf
    @test @inferred(logcdf(LogBNormal(0.0, 0.0), -0.5)) === -Inf

    @test @inferred(logcdf(LogBNormal(0.0, 0.0), 1.0f0)) === -0.0
    @test @inferred(logcdf(LogBNormal(0.0f0, 0.0f0), 1.0)) === -0.0
    @test @inferred(logcdf(LogBNormal(0.0f0, 0.0f0), 1.0f0)) === -0.0f0

    @test isnan_type(Float64, @inferred(logcdf(LogBNormal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(logcdf(LogBNormal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(logcdf(LogBNormal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(logcdf(LogBNormal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(logcdf(LogBNormal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(logcdf(LogBNormal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logcdf(LogBNormal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(logcdf(LogBNormal(0 // 1, 0 // 1), 1 // 1)) === -0.0
    @test @inferred(logcdf(LogBNormal(0 // 1, 0 // 1), 0 // 1)) === -Inf
    @test @inferred(logcdf(LogBNormal(0 // 1, 0 // 1), -1 // 1)) === -Inf
    @test isnan_type(Float64, @inferred(logcdf(LogBNormal(0 // 1, 0 // 1), NaN)))

    @test @inferred(logcdf(LogBNormal(0.0, 0.0), BigInt(1))) == big(0.0)
    @test @inferred(logcdf(LogBNormal(0.0, 0.0), BigInt(0))) == big(-Inf)
    @test @inferred(logcdf(LogBNormal(0.0, 0.0), BigInt(-1))) == big(-Inf)
    @test @inferred(logcdf(LogBNormal(0.0, 0.0), BigFloat(1))) == big(0.0)
    @test @inferred(logcdf(LogBNormal(0.0, 0.0), BigFloat(0))) == big(-Inf)
    @test @inferred(logcdf(LogBNormal(0.0, 0.0), BigFloat(-1))) == big(-Inf)
    @test isnan_type(BigFloat, @inferred(logcdf(LogBNormal(0.0, 0.0), BigFloat(NaN))))

    # ccdf
    @test @inferred(ccdf(LogBNormal(0.0, 0.0), 1.0)) === 0.0
    @test @inferred(ccdf(LogBNormal(0.0, 0.0), 0.5)) === 1.0
    @test @inferred(ccdf(LogBNormal(0.0, 0.0), 0.0)) === 1.0
    @test @inferred(ccdf(LogBNormal(0.0, 0.0), -0.5)) === 1.0

    @test @inferred(ccdf(LogBNormal(0.0, 0.0), 1.0f0)) === 0.0
    @test @inferred(ccdf(LogBNormal(0.0f0, 0.0f0), 1.0)) === 0.0
    @test @inferred(ccdf(LogBNormal(0.0f0, 0.0f0), 1.0f0)) === 0.0f0

    @test isnan_type(Float64, @inferred(ccdf(LogBNormal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(ccdf(LogBNormal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(ccdf(LogBNormal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(ccdf(LogBNormal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(ccdf(LogBNormal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(ccdf(LogBNormal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(ccdf(LogBNormal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(ccdf(LogBNormal(0 // 1, 0 // 1), 1 // 1)) === 0.0
    @test @inferred(ccdf(LogBNormal(0 // 1, 0 // 1), 0 // 1)) === 1.0
    @test @inferred(ccdf(LogBNormal(0 // 1, 0 // 1), -1 // 1)) === 1.0
    @test isnan_type(Float64, @inferred(ccdf(LogBNormal(0 // 1, 0 // 1), NaN)))

    @test @inferred(ccdf(LogBNormal(0.0, 0.0), BigInt(1))) == big(0.0)
    @test @inferred(ccdf(LogBNormal(0.0, 0.0), BigInt(0))) == big(1.0)
    @test @inferred(ccdf(LogBNormal(0.0, 0.0), BigInt(-1))) == big(1.0)
    @test @inferred(ccdf(LogBNormal(0.0, 0.0), BigFloat(1))) == big(0.0)
    @test @inferred(ccdf(LogBNormal(0.0, 0.0), BigFloat(0))) == big(1.0)
    @test @inferred(ccdf(LogBNormal(0.0, 0.0), BigFloat(-1))) == big(1.0)
    @test isnan_type(BigFloat, @inferred(ccdf(LogBNormal(0.0, 0.0), BigFloat(NaN))))

    # logccdf
    @test @inferred(logccdf(LogBNormal(0.0, 0.0), 1.0)) === -Inf
    @test @inferred(logccdf(LogBNormal(0.0, 0.0), 0.5)) === -0.0
    @test @inferred(logccdf(LogBNormal(0.0, 0.0), 0.0)) === -0.0
    @test @inferred(logccdf(LogBNormal(0.0, 0.0), -0.5)) === -0.0

    @test @inferred(logccdf(LogBNormal(0.0, 0.0), 1.0f0)) === -Inf
    @test @inferred(logccdf(LogBNormal(0.0f0, 0.0f0), 1.0)) === -Inf
    @test @inferred(logccdf(LogBNormal(0.0f0, 0.0f0), 1.0f0)) === -Inf32

    @test isnan_type(Float64, @inferred(logccdf(LogBNormal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(logccdf(LogBNormal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(logccdf(LogBNormal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(logccdf(LogBNormal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(logccdf(LogBNormal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(logccdf(LogBNormal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logccdf(LogBNormal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(logccdf(LogBNormal(0 // 1, 0 // 1), 1 // 1)) === -Inf
    @test @inferred(logccdf(LogBNormal(0 // 1, 0 // 1), 0 // 1)) === -0.0
    @test @inferred(logccdf(LogBNormal(0 // 1, 0 // 1), -1 // 1)) === -0.0
    @test isnan_type(Float64, @inferred(logccdf(LogBNormal(0 // 1, 0 // 1), NaN)))

    @test @inferred(logccdf(LogBNormal(0.0, 0.0), BigInt(1))) == big(-Inf)
    @test @inferred(logccdf(LogBNormal(0.0, 0.0), BigInt(0))) == big(0.0)
    @test @inferred(logccdf(LogBNormal(0.0, 0.0), BigInt(-1))) == big(0.0)
    @test @inferred(logccdf(LogBNormal(0.0, 0.0), BigFloat(1))) == big(-Inf)
    @test @inferred(logccdf(LogBNormal(0.0, 0.0), BigFloat(0))) == big(0.0)
    @test @inferred(logccdf(LogBNormal(0.0, 0.0), BigFloat(-1))) == big(0.0)
    @test isnan_type(BigFloat, @inferred(logccdf(LogBNormal(0.0, 0.0), BigFloat(NaN))))

    # quantile
    @test @inferred(quantile(LogBNormal(1.0, 0.0), 0.0f0)) === 0.0
    @test @inferred(quantile(LogBNormal(1.0, 0.0f0), 1.0)) === Inf
    @test @inferred(quantile(LogBNormal(1.0f0, 0.0), 0.5)) ===  exp(1)
    @test @inferred(quantile(LogBNormal(1.0f0, 0.0, 10.0), 0.5)) ===  exp10(1)
    @test isnan_type(Float64, @inferred(quantile(LogBNormal(1.0f0, 0.0), NaN)))
    @test @inferred(quantile(LogBNormal(1.0f0, 0.0f0), 0.0f0)) === 0.0f0
    @test @inferred(quantile(LogBNormal(1.0f0, 0.0f0), 1.0f0)) === Inf32
    @test @inferred(quantile(LogBNormal(1.0f0, 0.0f0), 0.5f0)) === exp(1.0f0)
    @test @inferred(quantile(LogBNormal(1.0f0, 0.0f0, 10.0f0), 0.5f0)) === exp10(1.0f0)
    @test isnan_type(Float32, @inferred(quantile(LogBNormal(1.0f0, 0.0f0), NaN32)))
    @test @inferred(quantile(LogBNormal(1//1, 0//1), 1//2)) === exp(1)
    @test @inferred(quantile(LogBNormal(1//1, 0//1, 10//1), 1//2)) === exp10(1)

    # cquantile
    @test @inferred(cquantile(LogBNormal(1.0, 0.0), 0.0f0)) === Inf
    @test @inferred(cquantile(LogBNormal(1.0, 0.0f0), 1.0)) === 0.0
    @test @inferred(cquantile(LogBNormal(1.0f0, 0.0), 0.5)) === exp(1)
    @test @inferred(cquantile(LogBNormal(1.0f0, 0.0, 10.0), 0.5)) === exp10(1)
    @test isnan_type(Float64, @inferred(cquantile(LogBNormal(1.0f0, 0.0), NaN)))
    @test @inferred(cquantile(LogBNormal(1.0f0, 0.0f0), 0.0f0)) === Inf32
    @test @inferred(cquantile(LogBNormal(1.0f0, 0.0f0), 1.0f0)) === 0.0f0
    @test @inferred(cquantile(LogBNormal(1.0f0, 0.0f0), 0.5f0)) === exp(1.0f0)
    @test @inferred(cquantile(LogBNormal(1.0f0, 0.0f0, 10.0f0), 0.5f0)) === exp10(1.0f0)
    @test isnan_type(Float32, @inferred(cquantile(LogBNormal(1.0f0, 0.0f0), NaN32)))
    @test @inferred(cquantile(LogBNormal(1//1, 0//1), 1//2)) === exp(1)
    @test @inferred(cquantile(LogBNormal(1//1, 0//1, 10//1), 1//2)) === exp10(1)

    # gradlogpdf
    @test @inferred(gradlogpdf(LogBNormal(0.0, 1.0), 1.0)) === -1.0
    @test @inferred(gradlogpdf(LogBNormal(0.0, 1.0), exp(-1))) === 0.0
    @test @inferred(gradlogpdf(LogBNormal(0.0, 1.0), 0.0)) === 0.0
    @test @inferred(gradlogpdf(LogBNormal(0.0, 1.0), -0.5)) === 0.0

    @test @inferred(gradlogpdf(LogBNormal(0.0, 1.0), 1.0f0)) === -1.0
    @test @inferred(gradlogpdf(LogBNormal(0.0f0, 1.0f0), 1.0)) === -1.0
    @test @inferred(gradlogpdf(LogBNormal(0.0f0, 1.0f0), 1.0f0)) === -1.0f0

    @test isnan_type(Float64, @inferred(logccdf(LogBNormal(0.0, 1.0), NaN)))
    @test isnan_type(Float64, @inferred(logccdf(LogBNormal(NaN, 1.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(logccdf(LogBNormal(NaN, 1.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(logccdf(LogBNormal(NaN, 1.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(logccdf(LogBNormal(NaN32, 1.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(logccdf(LogBNormal(NaN32, 1.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logccdf(LogBNormal(NaN32, 1.0f0), -1.0f0)))

    @test @inferred(gradlogpdf(LogBNormal(0 // 1, 1 // 1), 1 // 1)) === -1.0
    @test @inferred(gradlogpdf(LogBNormal(0 // 1, 1 // 1), 0 // 1)) === 0.0
    @test @inferred(gradlogpdf(LogBNormal(0 // 1, 1 // 1), -1 // 1)) === 0.0
    @test isnan_type(Float64, @inferred(gradlogpdf(LogBNormal(0 // 1, 1 // 1), NaN)))

    @test @inferred(gradlogpdf(LogBNormal(0.0, 1.0), BigInt(1))) == big(-1.0)
    @test @inferred(gradlogpdf(LogBNormal(0.0, 1.0), BigInt(0))) == big(0.0)
    @test @inferred(gradlogpdf(LogBNormal(0.0, 1.0), BigInt(-1))) == big(0.0)
    @test @inferred(gradlogpdf(LogBNormal(0.0, 1.0), BigFloat(1))) == big(-1.0)
    @test @inferred(gradlogpdf(LogBNormal(0.0, 1.0), BigFloat(0))) == big(0.0)
    @test @inferred(gradlogpdf(LogBNormal(0.0, 1.0), BigFloat(-1))) == big(0.0)
    @test isnan_type(BigFloat, @inferred(gradlogpdf(LogBNormal(0.0, 1.0), BigFloat(NaN))))
end
