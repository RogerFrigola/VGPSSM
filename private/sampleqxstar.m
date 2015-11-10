function Xfixedlag = sampleqxstar(S, tt, P, options)
% Use an SMC fixed-lag smoother to approximately sample from smoothing
% distribution. 

% Add extra edge data if necessary
numLeftEdge  = min(tt(1) - 1, options.minibatch.edgeLength);
numRightEdge = min(S.T - tt(end), options.minibatch.edgeLength);
tt = tt(1)-numLeftEdge:tt(end)+numRightEdge;

% Particle filter setup
numParticles = options.qxstar.numParticles;
numLagSteps = options.qxstar.numLagSteps;
X = nan(numel(tt)+1, options.numStates, numParticles);
U = S.U(tt,:); % Inputs for mini-batch
Y = S.Y(tt,:); % Data for mini-batch
Xfixedlag = nan(numel(tt), options.numStates, numParticles, 2); % 4-th dimension stores pairs x_{t-1}, x_t
W = nan(numel(tt), numParticles);
% Sample from p(x_0)
X(1,:,:) = repmat(options.px0{1}', [1 1 numParticles]); % mean of p(x_0) FIXME: when using minibatches we are not starting at x_0
X(1,:,:) = X(1,:,:) + permute( chol(options.px0{2} + 1e-6 * eye(size(options.px0{2})))' * ...
    randn(options.numStates,numParticles), [3 1 2]);
W(1,:) = 1 / numParticles;

% Convert natural parameters to mean and covariance
[mu, Sigma] = nat2mv(P.eta1new, P.eta2new);

% Normal SMC without looking at f
Kuu = options.gpf.covfun(P.theta.gp, P.Z);
L = chol(Kuu + 1e-6 * Kuu(1,1) * eye(size(Kuu)));
for i = 1:numel(tt)
    xu = [P321(X(i,:,:)) repmat(U(i,:), numParticles, 1)];
    % Predict
    Kmu = options.gpf.covfun(P.theta.gp, xu, P.Z);
    A = (Kmu / L) / L';  % Kmu / Kuu
    noise = randn(options.numStates, numParticles)' * chol(P.theta.Q);
    X(i+1,:,:) = P321( options.gpf.meanfun(xu) + (A*mu) + noise );
    % Compute weights
    V  = L' \ Kmu';
    % Also return input signal
    B = options.gpf.covfun(P.theta.gp, xu, 'diag') - sum(V.*V, 1)';  % Only diagonal sum(V.*V,1)' is like diag(Kmu*invKuu*Kmu')
    switch options.lik.type
        % This is the likelihood plus the extra exp term of the
        % auxiliary system
        case 'GaussianDiag'
            logw = logmvnpdf(Y(i,:), P321(X(i+1,:,:)), P.theta.lik.R) + TraceTerm(P.theta.Q, B, P321(A), Sigma);
        case 'Lin+GaussianDiag'
            % N( y | C * x, R)
            Cx = (P.theta.lik.C * P321(X(i+1,:,:))')';
            logw = logmvnpdf(Y(i,:), Cx, P.theta.lik.R) + TraceTerm(P.theta.Q, B, P321(A), Sigma);
    end
    W(i+1,:) = exp(logw - max(logw)); % Remove max to avoid numerical issues
    w = W(i+1,:) / sum(W(i+1,:));
    % Resample
    ii = resampling(w, 1);
    X(1:i+1,:,:) = X(1:i+1,:,ii);
    
    % Store fixed lag
    if i > numLagSteps
        Xfixedlag(i-numLagSteps,:,:,1) = X(i-numLagSteps,  :,:); % @ t
        Xfixedlag(i-numLagSteps,:,:,2) = X(i-numLagSteps+1,:,:); % @ t+1
    end
end
%disp([char(8) ', ESS: ' num2str(round(ess(w)))])
disp([char(8) ', Unique: ' num2str(numel(unique(X(i-numLagSteps,:,:))))])

% End of trajectories
Xfixedlag(i-numLagSteps+1:end,:,:,1) = X(i-numLagSteps+1:end-1,:,:);
Xfixedlag(i-numLagSteps+1:end,:,:,2) = X(i-numLagSteps+2:end,  :,:);
% Remove data at edges
Xfixedlag = Xfixedlag(1+numLeftEdge:end-numRightEdge,:,:,:);

if any(isnan(Xfixedlag(:)))
    keyboard
end

end



function trtrm = TraceTerm(Q, B, A, Sigma)
% Compute: -1/2 * trace(inv(Q) * (A * Sigma * A'))

[~,M,numParticles] = size(A);
D = size(Sigma, 1) / M;
Y = nan(D,D,numParticles);
% TODO could exploit symmetry and use some tensor product library
ii = 1:M;

if isdiag(Q)
    invQ = inv(Q);
    ASigmaAt = nan(numParticles, 1);
    trtrm = zeros(numParticles, 1);
    for d = 1:D
        ASigmaAt = sum((P321(A) * Sigma((d-1)*M+ii, (d-1)*M+ii)) .* P321(A), 2);
        trtrm = trtrm + invQ(d,d) * (B + ASigmaAt);
    end
    trtrm = -0.5 * trtrm;
else
    for i = 0:D-1
        for j = 0:D-1 
            for k = 1:numParticles
                Y(i+1,j+1,k) = A(1,:,k) * Sigma(i*M+ii, j*M+ii) * permute(A(1,:,k),[2 1 3]);
            end
            if i == j
                Y(i+1,j+1,:) = P321(B) + Y(i+1,j+1,:);
            end
            %Y(i+1,j+1,:) = bsxfun(@times, A, bsxfun(@times, Sigma(i*M+ii, j*M+ii), permute(A,[2 1 3])));
        end
    end
    % TODO could use some tensor product library
    invQ = inv(Q);
    for k = 1:numParticles
        Y(:,:,k) = invQ * Y(:,:,k);
    end
    % Compute trace
    trtrm = Y(1,1,:);
    for d = 2:D
        trtrm = trtrm + Y(d,d,:);
    end
    trtrm = -0.5 * trtrm;
    trtrm = P321(trtrm);
end


end




