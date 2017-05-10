function x = MRA_EM(X, sigma, x, tol, batch_niter)
% Expectation maximization algorithm for multireference alignment.
% X: data (each column is an observation)
% sigma: noise standard deviation affecting measurements
% x: initial guess for the signal (optional)
% tol: EM stops iterating if two subsequent iterations are closer than tol
%      in 2-norm, up to circular shift (default: 1e-5).
% batch_niter: number of batch iterations to perform before doing full data
%              iterations if there are more than 3000 observations
%              (default: 3000.)
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    % X contains M observations, each of length N
    [N, M] = size(X);
    
    % Initial guess of the signal
    if ~exist('x', 'var') || isempty(x)
        if isreal(X)
            x = randn(N, 1);
        else
            x = randn(N, 1) + 1i*randn(N, 1);
        end
    end
    x = x(:);
    assert(length(x) == N, 'Initial guess x must have length N.');
    
    % Tolerance to declare convergence
    if ~exist('tol', 'var') || isempty(tol)
        tol = 1e-5;
    end
    
    
    % In practice, we iterate with the DFT of the signal x
    fftx = fft(x);
    
    % Precomputations on the observations
    fftX = fft(X);
    sqnormX = repmat(sum(abs(X).^2, 1), N, 1);
    
    % If the number of observations is large, get started with iterations
    % over only a sample of the observations
    if M >= 3000
        
        if ~exist('batch_niter', 'var') || isempty(batch_niter)
            batch_niter = 3000;
        end
        batch_size = 1000;
        
        tic_batch = tic();
        
        for iter = 1 : batch_niter
            
            sample = randi(M, batch_size, 1);
            fftx_new = EM_iteration(fftx, fftX(:, sample), sqnormX(:, sample), sigma);
            
%             if relative_error(ifft(fftx), ifft(fftx_new)) < 100*tol
%                 break;
%             end
            
            fftx = fftx_new;
            
        end
        
        fprintf('\t\tEM: %d batch iterations, %.2g [s]\n', iter, toc(tic_batch));
        
    end
    
    tic_full = tic();
    
    % In any case, finish with full passes on the data
    full_niter = 10000;
    for iter = 1 : full_niter
        
        fftx_new = EM_iteration(fftx, fftX, sqnormX, sigma);

        if relative_error(ifft(fftx), ifft(fftx_new)) < tol
            break;
        end

        fftx = fftx_new;

    end
    
    fprintf('\t\tEM: %d full iterations, %.2g [s]\n', iter, toc(tic_full));
    
    
    x = ifft(fftx);

end


% Execute one iteration of EM with current estimate of the DFT of the
% signal given by fftx, and DFT's of the observations stored in fftX, and
% squared 2-norms of the observations stored in sqnormX, and noise level
% sigma.
function fftx_new = EM_iteration(fftx, fftX, sqnormX, sigma)

    C = ifft(bsxfun(@times, conj(fftx), fftX));
    T = (2*C - sqnormX)/(2*sigma^2);
    T = bsxfun(@minus, T, max(T, [], 1));
    W = exp(T);
    W = bsxfun(@times, W, 1./sum(W, 1));
    fftx_new = mean(conj(fft(W)).*fftX, 2);

end




        
%% Explicit old code: very slow but easier to parse (might be broken)
% fprintf('%d\n', iter);
% 
% x_new = zeros(N, 1);
% sample = 1:M;
% for m = sample
% 
%     % Get one of the observations
%     y = X(:, m);
%     yhat = fftX(:, m);
% 
%     % Compute squared ell_2 distance between x and all shifts of y
%     correlation = ifft(conj(fft(x)).*yhat);
%     sqdists = norm(x)^2 + norm(y)^2 - 2*correlation;
% 
%     % Use Bayes rule with a uniform prior to transform the
%     % distances into probabilities (actually, densities) for each
%     % possible shift. Take care of shifting the coefficients in the
%     % exponential to avoid IEEE complications.
%     t = -sqdists/(2*sigma^2);
%     w = exp(t - max(t));
%     w = w / sum(w);
% 
%     % Compute a weighted average of all shifted versions of y
%     avg = ifft(fft(y).*conj(fft(w));
% 
%     % The next estimator of x will be the average over observations
%     x_new = x_new + avg;
% 
% end
% x = x_new / M;