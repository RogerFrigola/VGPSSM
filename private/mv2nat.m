function [eta1, eta2] = mv2nat(mu, Sigma)
% Natural parameters of a Gaussian from mean and covariance

eta1 = Sigma \ mu(:);
eta2 = -0.5 * inv(Sigma);

eta1 = reshape(eta1, size(mu));