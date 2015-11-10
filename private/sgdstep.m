function [theta, state] = sgdstep(alg, theta, state, grad, hypers)

switch alg
    case 'Adam'
        % Kingma and Ba 2015, Adam: a Method for Stochastic Optimization
        state.t = state.t + 1;
        thisbeta1 = hypers.beta1 * hypers.lambda ^ (state.t - 1);
        state.m = thisbeta1 * state.m + (1 - thisbeta1) * grad;
        state.v = hypers.beta2 * state.v + (1 - hypers.beta2) * grad .^ 2;
        mhat = state.m / (1 - hypers.beta1 ^ state.t);
        vhat = state.v / (1 - hypers.beta2 ^ state.t);
        theta = theta - hypers.alpha * mhat ./ (sqrt(vhat) + hypers.epsilon);
        
    case 'RMSProp'
        % Tieleman, T. and Hinton, G. (2012), Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning
        state.r = (1 - hypers.gamma) * grad .^2 + hypers.gamma * state.r;
        v = hypers.alpha * grad ./ sqrt(state.r);
        theta = theta - v;
end

end