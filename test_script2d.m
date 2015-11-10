function test_script2d()

options.numStates = 2;
options.bDerivativeStates = false; % define state as x = (q, qdot)
options.optimisation.alg = 'RMSProp';
options.px0 = {zeros(options.numStates,1) zeros(options.numStates, options.numStates)}; % mean and cov of p(x_0)
options.qxstar.type = 'SMC'; % How to get q*(x_{0:T})
options.qxstar.numParticles = 100;
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
options.convergence.numIter = 200;
options.minibatch.type = 'Uniform';
options.minibatch.lengthMiniBatch = 50;
options.minibatch.edgeLength = 10;
options.inifun = @InitialiseParameters2d;
options.debugplot = @Plot2DSystem;

[Y, U] = generatedata();
vgpssm(Y, U, options)

end


function P = InitialiseParameters2d(S, options)
% TODO: update this has hard coded stuff and won not work in general!!!!

% Inducing inputs (spherical Gaussian + random selection of inputs)
P.Z = [ 3 * randn(options.gpf.numInducingPoints, options.numStates) ...
        S.U(randperm(size(S.U,1), Plot2DSystemoptions.gpf.numInducingPoints),:)];

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
        disp('N4SID C matrix: ')
        disp(sys.C)
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


function [Y, U] = generatedata()
% Generate data from nonlin system (no external inputs).

persistent X

T = 1e4;
U = randn(T,0);
if isempty(X)
    A = [0 1; -1 -1];
    B = [];
    C = eye(2);
    D = [];
    dt = 0.4;
    dsys = c2d(ss(A,B,C,D), dt);
    
    X = nan(T,2);
    X(1,:) = [1 1];
    for i = 2:T
        X(i,:) = (dsys.A * X(i-1,:)')' + randn(1,2);
    end
end
Y = X * [1 0]'; % no observation noise!

end


function Plot2DSystem(P,options)

[mu,Sigma] = nat2mv(P.eta1new, P.eta2new);

% Ground truth
%                x1      x2
%        x1  0.9306   0.321
%        x2  -0.321  0.6096
A = [0.9306   0.321;  -0.321  0.6096];
        

type = 'DoNotPlotMean';
switch type
    case 'PlotMean'
        m = options.gpf.meanfun(P.Z);
        subplot(121)
        plot3(P.Z(:,1), P.Z(:,2), m(:,1) + mu(:,1), 'bo')
        grid on; rotate3d on
        subplot(122)
        plot3(P.Z(:,1), P.Z(:,2), m(:,2) + mu(:,2), 'bo')
        grid on; rotate3d on
        
        xlims = get(gca,'XLim');
        ylims = get(gca,'YLim');
        [X1,X2] = meshgrid(linspace(xlims(1),xlims(2),10), linspace(ylims(1),ylims(2),10));
        subplot(121)
        hold on
        surf(X1, X2, reshape([X1(:) X2(:)] * A(1,:)', size(X1)),'FaceColor','none')
        subplot(122)
        hold on
        surf(X1, X2, reshape([X1(:) X2(:)] * A(2,:)', size(X1)),'FaceColor','none')
        
        
    case 'DoNotPlotMean'
        subplot(121)
        plot3(P.Z(:,1), P.Z(:,2), mu(:,1), 'bo')
        grid on; rotate3d on
        subplot(122)
        plot3(P.Z(:,1), P.Z(:,2), mu(:,2), 'bo')
        grid on; rotate3d on
        
        xlims = get(gca,'XLim');
        ylims = get(gca,'YLim');
        [X1,X2] = meshgrid(linspace(xlims(1),xlims(2),10), linspace(ylims(1),ylims(2),10));
        m = options.gpf.meanfun([X1(:) X2(:)]);
        subplot(121)
        hold on
        mesh(X1, X2, reshape([X1(:) X2(:)] * A(1,:)' - m(:,1), size(X1)),'EdgeColor',[0 0 0])
        subplot(122)
        hold on
        mesh(X1, X2, reshape([X1(:) X2(:)] * A(2,:)' - m(:,2), size(X1)),'EdgeColor',[0 0 0])
end
% Compute error metric
pred  = options.gpf.meanfun(P.Z) + mu;
truth = P.Z * A';
disp([char(8) ', err:' num2str( sqrt( sum(sum( (pred-truth).^2 )) ) )])
%disp([pred truth pred-truth])


% Plot prediction
[mu,Sigma] = nat2mv(P.eta1new, P.eta2new);
xstar = [X1(:) X2(:)];
Kuu = options.gpf.covfun(P.theta.gp, P.Z);
L = chol(Kuu + 1e-6 * Kuu(1,1) * eye(size(Kuu)));
Kmu = options.gpf.covfun(P.theta.gp, xstar, P.Z);
A = (Kmu / L) / L';
V  = L' \ Kmu';
B = options.gpf.covfun(P.theta.gp, xstar, 'diag') - sum(V.*V, 1)';
subplot(121)
hold on
mesh(X1, X2, reshape(A * mu(:,1), size(X1)),'FaceColor','none','EdgeColor',[0 0 1])
ASA = diag(A * Sigma(1:20,1:20) * A');
mesh(X1, X2, reshape(A * mu(:,1) + 2 * sqrt(B + ASA), size(X1)),'FaceColor','none','EdgeColor',[1 0 1])
mesh(X1, X2, reshape(A * mu(:,1) - 2 * sqrt(B + ASA), size(X1)),'FaceColor','none','EdgeColor',[1 0 1])
subplot(122)
hold on
mesh(X1, X2, reshape(A * mu(:,2), size(X1)),'FaceColor','none','EdgeColor',[0 0 1])
ASA = diag(A * Sigma(21:40,21:40) * A');
mesh(X1, X2, reshape(A * mu(:,2) + 2 * sqrt(B + ASA), size(X1)),'FaceColor','none','EdgeColor',[1 0 1])
mesh(X1, X2, reshape(A * mu(:,2) - 2 * sqrt(B + ASA), size(X1)),'FaceColor','none','EdgeColor',[1 0 1])

end