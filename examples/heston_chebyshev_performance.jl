"""
This Julia script is used to run the Heston implied volatility performance
tests in Julia directly.

The script depends on the function values on Chebyshev points beeing
available in the CSV files ./examples/values_jl_[Nd].csv.

Function values are calculated and stored via Python script
heston_chebyshev_performance.py.

It turns out there is no significant oerhead by using Julia from
Python via PyJulia calls.
"""


using DelimitedFiles

include("../src/multivariate_chebyshev_julia.jl")

perf_degrees = [ 1, 1, 2, 3 ]
for Nd in perf_degrees
    filename = string("./examples/values_jl_", string(Nd), ".csv")    
    values_jl = readdlm(filename)
    println("Read ", string(size(values_jl)), " data points from file ", filename)
    degrees = [ Nd for k in range(1,8) ]
    Z_jl = chebyshev_multi_points(degrees)
    #
    println("Calculate with batchmul")
    @time C_jl_bm = chebyshev_coefficients(degrees, Z_jl, values_jl[:,1])
    println("Calculate with matmul")
    @time C_jl_mm = chebyshev_coefficients(degrees, Z_jl, values_jl[:,1], matmul)
end
