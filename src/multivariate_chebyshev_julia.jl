
using LinearAlgebra

function cartesian_product(arrays)
    """
    Calculate the cartesian product of a list of input arrays.
    
    Parameters
    ----------
    arrays : 
        a parameter list of 1d arrays

    Returns
    -------
    matrix of size (N, D)
        a matrix with columns D equal to number of input arrays and
        rows N equal to the product of number of elements in input
        arrays

    """
    n_dims = size(arrays)[1]
    n_elms = prod([ size(array)[1] for array in arrays ])
    tuples = collect(Base.product(arrays...))
    tuples = reshape(tuples, n_elms)  # 1d array
    return [ tuples[i][j] for i in 1:n_elms, j in 1:n_dims ]
end

function batchmul(A, B)
    """
    Generalised matrix multiplication along first two dimensions.

    Based on
    https://stackoverflow.com/questions/57678890/batch-matrix-multiplication-in-julia
    """
    # first we need to check dimensions
    @assert length(size(A)) >= 2
    @assert length(size(A)) == length(size(B))
    @assert size(A)[2] == size(B)[1]
    for d in 3:length(size(A))
        @assert size(A, d)==1 || size(B, d)==1 || size(A, d)==size(B, d)
    end
    C = zeros(size(A, 1), size(B, 2), max.(size(A)[3:end],size(B)[3:end])...)
    for I in CartesianIndices(axes(C)[3:end])
        idx_A = min.(Tuple(I), size(A)[3:end])  # allow broadcasting for A
        idx_B = min.(Tuple(I), size(B)[3:end])  # allow broadcasting for B
        @views mul!(C[:, :, Tuple(I)...], A[:, :, idx_A...], B[:, :, idx_B...])
    end
    return C
end

function matmul(A, B)
    """
    Generalised matrix multiplication along last two dimensions.

    This mimics Numpy's matmul behaviour.
    """
    # first we need to check dimensions
    @assert length(size(A)) >= 2
    @assert length(size(A)) == length(size(B))
    @assert size(A)[end] == size(B)[end-1]
    for d in 1:length(size(A))-2
        @assert size(A, d)==1 || size(B, d)==1 || size(A, d)==size(B, d)
    end
    n_dims = length(size(A))
    C = zeros(max.(size(A)[1:end-2],size(B)[1:end-2])..., size(A, n_dims-1), size(B, n_dims))
    idxs_A = (axes(A,d) for d in 1:n_dims-2)
    idxs_B = (axes(B,d) for d in 1:n_dims-2)
    idxs_C = (axes(C,d) for d in 1:n_dims-2)
    for j in axes(B, n_dims)
        for i in axes(A, n_dims-1)
            for k in axes(A, n_dims)
                @views C[idxs_C...,i,j] += A[idxs_A...,i,k] .* B[idxs_B...,k,j]  # .* ensures proper broadcast
            end
        end
    end
    return C
end

function chebyshev_transform(x, a = -1.0, b = 1.0)
    """
    Transform input x from standard domain [-1, 1] to general
    hyper-rectangular domain.

    Parameters
    ----------
    x : 1d array or 2d array on the standard domain [-1, 1]
    a : float or 1d array
        lower boundary of domain, if a is array then we require size(a)[1]
        equal to size(x)[-1]
    b : float or 1d array
        upper boundary of domain, if b is array then we require size(b)[1]
        equal to size(x)[-1]
    
    Returns
    -------
    1d array or 2d array of size size(x)    
    """
    return a .+ 0.5 .* (b .- a) .* (x .+ 1.0)
end

function chebyshev_inverse_transform(y, a = -1.0, b = 1.0)
    """
    Transform input y from general hyper-rectangular domain to
    standard domain [-1, 1].

    Parameters
    ----------
    y : 1d array or 2d array on general hyper-rectangular domain
    a : float or 1d array
        lower boundary of domain, if a is array then we require size(a)[1]
        equal to size(x)[-1]
    b : float or 1d array
        upper boundary of domain, if b is array then we require size(b)[1]
        equal to size(x)[-1]
    
    Returns
    -------
    1d array or 2d array of size size(x)    
    """
    return 2 ./ (b.-a) .* (y.-a) .- 1.0
end

function chebyshev_points(degrees)
    """
    Calculate the Chebyshev points of second kind used for interpolation.

    Parameters
    ----------
    degrees : list of int
        each entry represents the maximum polynomial degree per dimension,
        size(degrees)[1] corresponds to the number of dimensions

    Returns
    -------
    list of 1d arrays


    Chebyshev points of second represent the extrema of Chebyshev polynomials
    of first kind.
    """
    return [ cos.(pi .* collect(range(0,stop=1,length=n+1))) for n in degrees ]
end

function chebyshev_multi_points(degrees)
    """
    Calculate multivariate Chebyshev points on a standard domain [-1,1].
    
    Parameters
    ----------
    degrees : list of int
        each entry represents the maximum polynomial degree per dimension,
        len(degrees) corresponds to the number of dimensions
        
    Returns
    -------
    2d array of shape (N,D)
        number of rows N equals product over all (N_d + 1) where N_d is
        the maximum polynomial degree in dimension d (d=1...D) and D is
        the number of dimensions of the tensor.
    """
    points = chebyshev_points(degrees)
    multi_points = cartesian_product(points)
    return multi_points
end

function chebyshev_polynomials(x, max_degree)
    """
    Calculate the Chebyshev polynomials T_0(x), ..., T_N(x)
    for up to maximum degree N.

    Parameters
    ----------
    x : float or 1d array
        for interpolation x is assumed to be in [-1, 1] (element-wise)

    Returns
    -------
    1d array if input is float or 2d array if input is array. First
    dimension represents maximum degree N.
    """
    T = [ 1.0 .+ 0 .* x ]
    if max_degree==0
        return hcat(T...)
    end
    push!(T, x)
    if max_degree==1
        return hcat(T...)
    end
    for d in 2:max_degree
        push!(T, 2*x.*T[end]-T[end-1])
    end
    return hcat(T...)
end

function chebyshev_batch_call(C, x)
    """
    Calculate

    z = [...[C * T(x_D)] * ...] * T(x_1)

    for a tensor C and input points x.

    T(x_d) are Chebyshev polynomials to degree N_d.

    Parameters
    ----------
    C : ndarray with suitable shape
        a tensor specifying the maximum polynomial degrees per dimension.

    x : ndarray with shape (N, D)
        a matrix where first dimension N represents batch size and
        second dimension D represents number of dimensions of tensor.
        
    Returns
    -------
    1d array of shape (N,)

    This is the basic operation for calibration and and interpolation.

    Re-shaping is to ensure proper broadcast in multiplication.
    """
    degrees = [ d-1 for d in size(C) ]
    @assert length(size(x))==2
    @assert size(x, 2)==length(degrees)
    res = reshape(C, (size(C)..., 1))
    for d in 1:size(x, 2)
        T = chebyshev_polynomials(x[:,d], degrees[d])
        shape = append!([1], [size(T,2)], [ 1 for k in 1:length(size(res))-3 ], [size(T,1)])
        T = reshape(transpose(T), Tuple(shape))
        if length(size(res))==2  # last iteration
            res = reshape(res, (size(res, 1), 1, size(res, 2)))
        end
        res = batchmul(T, res)
        res = reshape(res, size(res)[2:end])
    end
    return reshape(res, (size(res)[end],))
end

function chebyshev_coefficients(degrees, multi_points, values)
    """
    Calculate coefficients of Chebyshev basis functions.

    Parameters
    ----------
    degrees : list of int
        each entry represents the maximum polynomial degree per dimension,
        len(degrees) corresponds to the number of dimensions
    multi_points : 2d array
        multivariate Chebyshev points on a standard domain [-1,1]
    values : 1d array
        the target function values for each D-dimensional multivariate
        Chebyshev point on the transformed domain of the tagret function.
        
    Returns
    -------
    ndarray of shape (degrees + 1)
    """
    @assert length(size(multi_points)) == 2
    @assert length(size(values)) == 1
    @assert size(multi_points, 1) == prod([n+1 for n in degrees])
    @assert size(multi_points, 1) == size(values, 1)
    @assert size(multi_points, 2) == length(degrees)
    #
    idx_list = [ 1:n+1 for n in degrees ]
    multi_idx = cartesian_product(idx_list)
    inner_node_factor = sum((1 .< multi_idx) .&& (multi_idx .< (degrees' .+ 1)), dims=2)
    values_adj = values .* 0.5.^(length(degrees) .- inner_node_factor)
    values_adj = reshape(values_adj, Tuple([n+1 for n in degrees])) # as tensor
    multi_coeff = chebyshev_batch_call(values_adj, multi_points)
    multi_coeff = multi_coeff .* (2 .^ inner_node_factor) ./ prod(degrees)
    coeff = reshape(multi_coeff, Tuple([n+1 for n in degrees])) # as tensor
    return coeff
end

function chebyshev_interpolation(y, coeff, a = -1.0, b = 1.0)
    """
    Calcuate multivariate Chebyshev interpolation.

    Parameters
    ----------
    y : 2d array
        Inputs of shape (N, D) where N is batch size and D is the number
        of dimensios. Input is assumed from general hyper-rectangular
        domain
    coeff : ndarray
        calibrated Chebyshev tensor coefficients
    a : float or 1d array
        lower boundary of domain, if a is array then we require a.shape[0]
        equal to y.shape[-1]
    b : float or 1d array
        upper boundary of domain, if b is array then we require b.shape[0]
        equal to y.shape[-1]
    
    Returns
    -------
    1darray of shape (N,)
    """
    x = chebyshev_inverse_transform(y, a, b)
    return chebyshev_batch_call(coeff, x)
end
