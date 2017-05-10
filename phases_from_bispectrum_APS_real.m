function z = phases_from_bispectrum_APS_real(B, z0)
% Iterative phase synchronization to estimate the phases of the DFT of a
% signal x given an estimator for its bispectrum, B. Input z0 if specified,
% provides the phase of the mean of x, as a unit-modulus complex number.
%
% Requires Manopt: http://www.manopt.org/
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    N = size(B, 1);
    assert(size(B, 2) == N, 'B must be square.');
    
    % For real signals, B is supposed to be Hermitian.
    B = (B+B')/2;

    if ~exist('z0', 'var') || isempty(z0)
        z0 = sign(B(1, 1));
    end
    if z0 == 0
        z0 = 1;
    end
    
    % Useful functions to enforce symmetry of phases of DFT of real signals
    down = @(u) u;
    up = @(u) u([1 ; (N:-1:2)']);
    downup = @(u) (down(u) + conj(up(u)))/2;
    
    % Random initial guess + enforce symmetry and z0, z1.
    z = randn(N, 1) + 1i*randn(N, 1);
    z(1) = sign(z0);
    z = sign(downup(z));
    
    % We want to find z such that Mfun(z) = z*z'.
    Mfun = @(z) B .* conj(circulant(z));
    
    % Run a fixed number of iterations (could be improved)
    for K = 1 : 15
        
        % Synchronization matrix: we want to maximize z'*M*z
        M = Mfun(z);
        
        % Solve the synchronization problem with Manopt
        options.verbosity = 0;
        options.tolgradnorm = 1e-6;
        problem.M = complexcirclefactory(N);
        problem.cost  = @(z) -real(z'*M*z)/(N^2);
        problem.egrad = @(z) -(2/N^2) * M*z;
        problem.ehess = @(z, zdot) -(2/N^2) * M*zdot;
        z = trustregions(problem, z, options);
        
        %  Fix global phase ambiguity by imposing sign of DC component
        z = z * (sign(z0) / z(1));
        % Impose symmetry proper for real signals
        z = sign(downup(z));
        
    end

end
