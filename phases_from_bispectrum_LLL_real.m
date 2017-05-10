function z = phases_from_bispectrum_LLL_real(B, z0, z1, ell)
%PHASE_RECOVERY_LLL: Recover the phase of the fourier transform of signal x
%from matrix of normalized bispectrum B.
%
%==========================================================================
%Input: B, n+1*n+1 matrix, normalized bispectrum of Y. Y is the noised
%       observation of signal x.
%Output:phi, n+1 - vector, the angle of x's fourier transform.
%
%==========================================================================
%We first calculate the angle of bispectrum beta, then solve the minimizing
%problem:
%                  min_{phi,k} ||A*phi-beta-2*pi*k||_{L1}        (1)
%
%In order to solve the problem, we first find a  matrix C with orthogonal
%rows, which satisfies C*A=0. Hence, we can solve k by
%                  min_{k} ||2*pi*C*k+C*beta||_{L2}              (2)
%We use the LLL algorithm to solve this problem, then put the result k into
%(1), and solve (1) for phi.
%
%==========================================================================
%Problem (2) is solved by codes from Xiao-Wen Chang and Tianyang Zhou
%Problem (1) is solved by cvx
%
%==========================================================================
%Coder: Chao Ma
%Date: 12-06-2016
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    % Make sure the MILES package is available.
    if ~exist('sils', 'file')
        addpath('MILES');
    end
    
    % ell = 1 or 2
    if ~exist('ell', 'var') || isempty(ell)
        ell = 1;
    end
    assert(ell == 1 || ell == 2, 'ell = 1 or 2');
    

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
    if z0 == 0
        z0 = 1;
    end
    if z1 == 0
        z1 = 1;
    end

    % Parameter preparation

    n = size(B, 1) - 1;
    
    % As x is a real signal, let X be its fourier transform, then
    % X(k)=conj(X(n+1-k)). So we only need to deal with half of the
    % frequencies.
    even = (mod(n,2)==0);
    if even
        nn = n/2;
    else
        nn = (n-1)/2;
    end

    % Save the index involved in calculating each bispectrum
    index=[];
    for i=1:floor(nn/2)
        for j=i:nn-i
            index=[index,[i;j]];
        end
    end
    N = size(index,2);

    % Get the angle of bispectrums we need
    beta=zeros(n, 1);
    for i=1:N
        beta(i)=angle(B(index(1,i)+1,index(1,i)+index(2,i)+1));
    end

    % Calculate matrix A
    A=zeros(N, nn);
    for i=1:N
        A(i,index(1,i))=A(i,index(1,i))+1;
        A(i,index(2,i))=A(i,index(2,i))+1;
        A(i,index(1,i)+index(2,i))=A(i,index(1,i)+index(2,i))-1;
    end

    % Calculate C that C*A=0. But here C is an integer matrix
    C=zeros(N-nn+1, N);
    K=zeros(nn);
    k=0;
    for i=1:floor(nn/2)
        for j=i:nn-i
            k=k+1;
            K(i,j)=k;
        end
    end
    k=0;
    for i=2:floor(nn/2)
        for j=i:nn-i
            k=k+1;
            C(k,[K(i,j),K(i-1,j+1),K(1,j),K(1,i-1)])=[1,-1,-1,1];
        end
    end

    % Turn C into a matrix with orthogonal rows
    C=orth(C')';

    % Solve the integer vector k by the LLL algorithm. To avoid the
    % underdetermined problem, simply set the first several entries of k to 0.
    b=-C*beta/2/pi;
    K=zeros(N,1);
    K(nn:N)=sils(C(:,nn:N),b,1);

    % Solve the angle vector phi by l1-minimization
    warning ('off');
    cvx_begin quiet
        variable phi(nn);
        minimize(norm(A*phi-(beta+2*pi*K), ell))
        subject to
            phi(1) == angle(z1)
    cvx_end
    warning('on');

    % Generate the full length phase vector
    if even
        phi=[angle(z0);phi;flipud(-phi)];
    else
        phi=[angle(z0);phi;0;flipud(-phi)];
    end

    z = exp(1j*phi);

end
