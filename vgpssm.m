function vgpssm(Y,U,options)
%VGPSSM
%
% Roger Frigola, 2014-2015.

S.T = size(Y,1);
S.numObs = size(Y,2);
if nargin < 2, U = zeros(S.T,0); end
S.numInputs = size(U,2);
if nargin < 3, options = DefaultOptions(); end

% Separately whiten inputs and outputs
%[S.Y, S.whit.muY, S.whit.RY] = whiten(Y);
%[S.U, S.whit.muU, S.whit.RU] = whiten(U);
S.Y = Y;
S.U = U;

figure

i = 1; % iteration number
P = options.inifun(S, options);
clear SGD; % clears persistent state
while ~IsConverged(i, options)
    disp(['Iter: ' num2str(i)])
    % Sample mini-batch
    [tt, numMiniBatches] = sampleminibatch(S.T, options);
    
    % Get samples from q*(x_{0:T})
    switch options.qxstar.type
        case {'SMC' 'SMC_FixedLagSmoother'}
            % Use SMC fixed-lag smoother
            qxstar = sampleqxstar(S, tt, P, options);
        case 'Variational'
            % TODO: parameterised q(x)
    end
    
    % Get q*(u)
    P = GetOptimalqu(P, qxstar, S.U(tt,:), numMiniBatches, options);
    
    % Get parameter gradient (hyperparameters, pseudo-inputs, etc.)
    gradP = GetGradient(P, qxstar, S.U(tt,:), S.Y(tt,:), options);
    
    % Update q(u)
    if i < 20, rho = 0.8; else rho = 0.4 * (options.convergence.numIter-i)/options.convergence.numIter; end
    disp([char(8) ', rho: ' num2str(rho)])
    P.eta1new = rho * P.eta1new + (1-rho) * P.eta1old;
    P.eta2new = rho * P.eta2new + (1-rho) * P.eta2old;
    
    % Update parameters
    if i < 50, gamma = 0; else gamma = 0.006 * (options.convergence.numIter-i)/options.convergence.numIter; end
    P.theta.lik.C = P.theta.lik.C + gamma * gradP.theta.lik.C;
    subplot(122); plot(i,P.theta.lik.C(1),'.'); hold on;
    %P = SGD(P, gradP, i, options);
    
    % DEBUG
    if true
        subplot(121)
        options.debugplot(P,options); drawnow
    end

    % Iteration number
    i = i + 1;
end

figure
options.debugplot(P,options)
keyboard

% Unwhiten
error('see p. 48')

end



% ----------------------------------------------------------------------- %
function options = DefaultOptions()
% Default options for vgpssm.m

options.numStates = 2;
options.bDerivativeStates = false; % define state as x = (q, qdot)
options.optimisation.alg = 'RMSProp';
options.px0 = {zeros(options.numStates,1) zeros(options.numStates, options.numStates)}; % mean and cov of p(x_0)
options.qxstar.type = 'SMC'; % How to get q*(x_{0:T})
options.qxstar.numParticles = 500;
options.qxstar.numLagSteps = 3;
options.gpf.meanfun = @(x) x(:, 1:options.numStates);
options.gpf.covfun  = @covMaternard3;
options.gpf.theta0  = [];
options.gpf.numInducingPoints = 20;
options.q.type = 'Diagonal';
options.lik.type = 'Lin+GaussianDiag'; % GP+Gaussian, Student, function handle, etc.
options.lik.theta0 = [];
options.ini.strategy = 'LinearSubSpace';
options.convergence.type = 'FixedIter';
options.convergence.numIter = 100;
options.minibatch.type = 'Uniform';
options.minibatch.lengthMiniBatch = 200;
options.minibatch.edgeLength = 10;
options.inifun = @DefaultInitialiseParameters;

end


function P = DefaultInitialiseParameters(S, options)
% TODO: update this has hard coded stuff and won not work in general!!!!

% Inducing inputs (spherical Gaussian + random selection of inputs)
P.Z = [ 3 * randn(options.gpf.numInducingPoints, options.numStates) ...
        S.U(randperm(size(S.U,1), options.gpf.numInducingPoints),:)];

% GP covfun hyper-parameters
if ~isempty(options.gpf.theta0)
    P.theta.gp = options.gpf.theta0;
else
    sigmaf = 1.6;
    P.theta.gp = [ 2.3 + zeros(size(P.Z, 2), 1); log(sigmaf)];
end

% State transition noise
P.theta.Q = eye(options.numStates);

% Dynamics initialisation
switch options.ini.strategy
    % Sample initial mean q(u) from prior p(u)
    case 'LinearSubSpace'
        % Use linear subspace identification method for initialisation
        data = iddata(S.Y, S.U);
        sys = n4sid(data, options.numStates);
        % convert to discrete ss model!!!!!!!
        mu = sys.A * P.Z(:,1:options.numStates)' + sys.B * P.Z(:,options.numStates+1:end)';
        mu = mu' - options.gpf.meanfun(P.Z);
    case 'RandomNonlinear'
        % TODO
    case 'MarginallyStableLinear'
        % TODO
end

% Likelihood
P.theta.lik = options.lik.theta0;
switch options.lik.type
    case 'Lin+GaussianDiag'
        if options.numStates >= size(S.Y, 2)
            P.theta.lik.C = [eye(size(S.Y, 2))  zeros(size(S.Y, 2), options.numStates-size(S.Y, 2))];
        end
end
% TODO: This should be initialised with the noise estimated when
% initialising the dynamics.
P.theta.lik.R = eye(size(S.Y, 2));

% Initialisation of Sigma will have an influence on smoother
%tmp = feval(options.gpf.covfun, P.theta.gp, P.Z);
tmp = 1^2 * eye(options.gpf.numInducingPoints);
Sigma = blkdiagn(tmp, options.numStates);
[P.eta1new, P.eta2new] = mv2nat(mu, Sigma);

end



function P = SGD(P, gradP, etaOpt, i, options)
% Wrapper for a stochastic gradient descent algorithm.

persistent state
if isempty(state)
    
end

[theta, state] = sgdstep(options.optimisation.alg, theta, state, grad, hypers);
end



function bConverged = IsConverged(i, options)
% Convergence test for stochastic variational inference

bConverged = 0;
switch options.convergence.type
    case 'FixedIter'
        if i > options.convergence.numIter
            bConverged = 1;
        end
    otherwise
        error('Convergence criterion not recognised.')
end

end



function P = GetOptimalqu(P, qxstar, U, numMiniBatches, options)
% Computes optimal q(u) using samples of q*(x)

P.eta1old = P.eta1new;
P.eta2old = P.eta2new;

numParticles = size(qxstar,3);
M   = options.gpf.numInducingPoints;
Kuu = options.gpf.covfun(P.theta.gp, P.Z);
invKuu = inv(Kuu + 1e-6 * eye(size(Kuu))); % FIXME: should only use L
L = chol(Kuu + 1e-6 * Kuu(1,1) * eye(size(Kuu)));

sum1 = zeros(numel(P.eta1new), 1);
sum2 = zeros(size(P.eta2new));
% TODO: ideally, this loop should be avoided
for i = 1:size(qxstar,1)
    xu  = [P321(qxstar(i,:,:,1)) repmat(U(i,:), numParticles, 1)];
    m   = options.gpf.meanfun(xu);
    Kmu = options.gpf.covfun(P.theta.gp, xu, P.Z);
    A = (Kmu / L) / L';  % Kmu / Kuu
    
    % Monte Carlo estimates of the expectations
    %sum1 = sum1 + 1/numParticles * sum(bsxfun(@times, P321(qxstar(i,:,:,2)), A))';
    tmp = P.theta.Q \ (P321(qxstar(i,:,:,2)) - m)';
    for d = 1:options.numStates
        ii = 1+(d-1)*M : d*M;
        sum1(ii) = sum1(ii) + mean(bsxfun(@times, A', tmp(d,:)), 2);
    end
        
    %sum2 = sum2 + 1/numParticles * (A'*A);
    if ~isdiag(P.theta.Q)
        error('TODO: implement for nondiagonal Q');
    else
        invQ = inv(P.theta.Q);
        tmp = 1/numParticles * (A'*A);
        for d = 1:options.numStates
            ii = 1+(d-1)*M : d*M;
            for e = 1:options.numStates
                if d ~= e, continue; end
                jj = 1+(e-1)*M : e*M;
                sum2(ii,jj) = sum2(ii,jj) + invQ(d,d) * tmp;
            end
        end
    end
end

% eta1 = theta.sqrtQ^-2 * numMiniBatches * sum1;
% eta2 = -0.5 * (invKuu + numMiniBatches * sum2 * theta.sqrtQ^-2);

P.eta1new = numMiniBatches * sum1 + blkdiagn(invKuu, options.numStates) * reshape(options.gpf.meanfun(P.Z),[],1);
P.eta1new = reshape(P.eta1new, [], options.numStates); 
P.eta2new = -0.5 * (blkdiagn(invKuu, options.numStates) + numMiniBatches * sum2);

end



function gradP = GetGradient(P, qxstar, U, Y, options)
% Gradient of the ELBO with respect to various parameters

gradP = P;

% Reuse Monte Carlo estimates of expectations from qstaru
[mu, Sigma] = nat2mv(P.eta1new, P.eta2new);
Sigma = stacksigma(Sigma, options.numStates);
Kuu    = options.gpf.covfun(P.theta.gp, P.Z);
invKuu = inv(Kuu + 1e-6 * eye(size(Kuu))); % FIXME: should only use L
L = chol(Kuu + 1e-6 * Kuu(1,1) * eye(size(Kuu)));
alpha = Kuu \ (mu - options.gpf.meanfun(P.Z));

% Pre-compute all necessary expectations wrt the smoothing distribution
phi6 = zeros(size(Y,2), 1);
phi7 = zeros(size(Y,2), numel(P.theta.lik.C));
% TODO: ideally, this loop should be avoided
numParticles = options.qxstar.numParticles;
for i = 1:size(qxstar,1)
    xu  = [P321(qxstar(i,:,:,1)) repmat(U(i,:), numParticles, 1)];
    m   = options.gpf.meanfun(xu);
    Kmu = options.gpf.covfun(P.theta.gp, xu, P.Z);
    A = (Kmu / L) / L';  % Kmu / Kuu
        
    % phi 6
    ztild = bsxfun(@minus, Y(i,:)', P.theta.lik.C * permute(qxstar(i,:,:,2), [2 3 1]) );
    phi6 = phi6 + mean(ztild .* ztild, 2);
    
    % phi 7
    for j = 1:numel(P.theta.lik.C)
        dCdtheta = zeros(size(P.theta.lik.C));
        dCdtheta(j) = 1;
        phi7(:,j) = phi7(:,j) + mean(ztild .* (dCdtheta * permute(qxstar(i,:,:,2), [2 3 1])), 2);
    end
    
%     % Monte Carlo estimates of the expectations
%     %sum1 = sum1 + 1/numParticles * sum(bsxfun(@times, P321(qxstar(i,:,:,2)), A))';
%     tmp = P.theta.Q \ (P321(qxstar(i,:,:,2)) - m)';
%     for d = 1:options.numStates
%         ii = 1+(d-1)*M : d*M;
%         sum1(ii) = sum1(ii) + mean(bsxfun(@times, A', tmp(d,:)), 2);
%     end
%         
%     %sum2 = sum2 + 1/numParticles * (A'*A);
%     if ~isdiag(P.theta.Q)
%         error('TODO: implement for nondiagonal Q');
%     else
%         invQ = inv(P.theta.Q);
%         tmp = 1/numParticles * (A'*A);
%         for d = 1:options.numStatesfunction P = SGD(P, gradP, etaOpt, i, options)
% Wrapper for a stochastic gradient descent algorithm.
%             ii = 1+(d-1)*M : d*M;
%             for e = 1:options.numStates
%                 if d ~= e, continue; end
%                 jj = 1+(e-1)*M : e*M;
%                 sum2(ii,jj) = sum2(ii,jj) + invQ(d,d) * tmp;
%             end
%         end
%     end
end
% TODO: normalise by num of minibatches as we do for eta1 and eta2


% Inducing inputs: P.Z
% dK
% gradP.Z = AssembleGrad();

% GP hyper-parameters: P.theta.gp
for i = 1:numel(P.theta.gp)
    tmp = 0;
    dKdtheta = options.gpf.covfun(P.theta.gp, P.Z, [], i);
    % Prior KL term
    for j = 1: options.numStates
        alphaalpha = alpha(:,j) * alpha(:,j)';
        tmp = tmp + 0.5 * trace(dKdtheta * (alphaalpha + invKuu * (Sigma(:,:,j) - eye(options.gpf.numInducingPoints)) ));
    end
    % Trace term
    dAdtheta = [];
    dBdtheta = [];
    
    % Dynamics term
    
    
    gradP.theta.gp(i) = tmp;
end

% State noise: P.theta.Q
% Trace term

% Dynamics term

gradP.theta.Q = [];


% Linear likelihood: P.theta.lik.C
% Data term
for i = 1:numel(P.theta.lik.C)
    gradP.theta.lik.C(i) = diag(inv(P.theta.lik.R))' * phi7(:,i);
end

% Linear likelihood: P.theta.lik.R, TODO: should reparameterise this!
% Data term
invR = inv(P.theta.lik.R);
for i = 1:size(P.theta.lik.R,1)
    dRdtheta = zeros(size(invR));
    dRdtheta(i,i) = 1;
    gradP.theta.lik.R(i,i) = - 0.5 * trace(invR * dRdtheta(i,i)) ...
                             + 0.5 * diag(invR * dRdtheta * invR)' * phi6;
end

end

