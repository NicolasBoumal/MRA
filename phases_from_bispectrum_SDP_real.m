function z = phases_from_bispectrum_SDP_real(B, z0, z1)
% SDP relaxation to estimate the phases of the DFT of a signal x given
% an estimator for its bispectrum, B. Entries z0 and z1, if specified,
% provide the phases of the first two entries of the DFT of x, as
% unit-modulus complex numbers (z0 is the phase of the mean, z1 is the
% phase of the first frequency.)
%
% Requires CVX: http://cvxr.com/cvx/
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
    if ~exist('z1', 'var') || isempty(z1)
        z1 = 1;
    end
    
    z0 = sign(z0);
    z1 = sign(z1);
    if z0 ~= 1 && z0 ~= -1
        z0 = 1;
    end
    if z1 == 0
        z1 = 1;
    end
    
    warning('off');
    
    cvx_precision high;
    cvx_begin sdp quiet
    variable Z(N, N) hermitian
    variable z(N) complex

        % Owing to the symmetries of z (due to the signal being real), we
        % have toeplitz(conj(z)) == circulant(conj(z));
        %
        % Notice the weights.
        Btilde = sign(B);
        W = abs(B);
        minimize ( norm( W .* (Z - Btilde .* toeplitz(conj(z))), 'fro') );

        subject to

            [Z z ; z' 1] >= 0; % positive semidefiniteness constraint

            diag(Z) == 1;

            z(1) == z0;
            z(2) == z1;
            z(N) == conj(z1);
            z(3:end-1) == conj(z((end-1):-1:3));

    cvx_end

    warning('on');
    
    % Ensure the constraints are satisfied exactly by projecting.
    z(1) = z0;
    z(2) = z1;
    z(N) = conj(z1);
    z(2:end) = (z(2:end) + conj(flipud(z(2:end))))/2;
    
    % Force the unit modulus constraints.
    z = sign(z);
    z(z == 0) = 1;

end
