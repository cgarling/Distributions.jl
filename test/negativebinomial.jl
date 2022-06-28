using Distributions
using ChainRulesTestUtils
using FiniteDifferences
using Test, ForwardDiff

# Currently, most of the tests for NegativeBinomial are in the "ref" folder.
# Eventually, we might want to consolidate the tests here

mydiffp(r, p, k) = r/p - k/(1 - p)

@testset "NegativeBinomial r=$r" for r in exp10.(range(-10, stop=2, length=25))
    # avoid p==1 since it's not differentiable
    @testset "p = $p" for p in exp10.(-10:0) .- eps()
        @testset "k = $k" for k in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
            @test ForwardDiff.derivative(_p -> logpdf(NegativeBinomial(r, _p), k), p) ≈ mydiffp(r, p, k) rtol=1e-12 atol=1e-12

            dist = NegativeBinomial(r, p)
            if p == 1. - eps()
                # Did not get to work:
                # test_rrule(logpdf, dist, k; fdm = backward_fdm(5, 1, max_range = r/2), rtol = 1e-5, atol = 1e-1)
            else
                test_rrule(logpdf, dist, k; fdm = central_fdm(5, 1, max_range = min(r, p)/2), rtol = 1e-10, atol = 1e-1)
            end
        end
    end
end

@testset "Check the corner case p==1" begin
    @test logpdf(NegativeBinomial(0.5, 1.0), 0) === 0.0
    @test logpdf(NegativeBinomial(0.5, 1.0), 1) === -Inf

    @testset "r=$r" for r in exp10.(range(-10, stop=2, length=25))
        test_rrule(logpdf, NegativeBinomial(r, 1.0), 0; fdm = backward_fdm(5, 1, max_range = r/2), rtol = 1e-8, atol = 1e-4)
    end
end
