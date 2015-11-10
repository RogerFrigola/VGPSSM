function [mu, Sigma] = nat2mv(eta1, eta2)
% Mean and covariance of a Gaussian from natural parameters 

Sigma = inv( -2 * eta2 );
mu = Sigma * eta1(:);

mu = reshape(mu, size(eta1));