function test_sgdstep()

theta = [50; 2];

% Adam hyper-parameters
hypers.alpha = 6e-1;
hypers.beta1 = 0.9;
hypers.beta2 = 0.999;
hypers.epsilon = 1e-8;
hypers.lambda = 1 - 1e-8;
% Initialise state
state.t = 0;
state.m = zeros(size(theta));
state.v = state.m;


% Optimise for a fixed number of iterations
numIter = 1000;
figure; hold on
f = nan(numIter,1);
theta = [theta, nan(size(theta,1), numIter)];
for t = 1:numIter
    [f(t),grad] = onehump(theta(:,t));
    [theta(:,t+1), state] = sgdstep('Adam', theta(:,t), state, grad, hypers);
end
plot(theta(1,:), theta(2,:), 'k.-')


% RMSProp 
clear hyper state
hypers.alpha = .1;
hypers.gamma = 0.9;
state.r = zeros(size(theta,1), 1);
for t = 1:numIter
    [f(t),grad] = onehump(theta(:,t));
    [theta(:,t+1), state] = sgdstep('RMSProp', theta(:,t), state, grad, hypers);
end
plot(theta(1,:), theta(2,:), 'r.-')

end

