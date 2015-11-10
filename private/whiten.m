function [Z, mu, R] = whiten(X)
% Whiten matrix X where each row is a datapoint so that: Z = (X-mu) * R
% where Z has zero mean and identity covariance matrix.

mu = mean(X);
Z  = bsxfun(@minus, X, mu);

Sigma   = cov(Z);
[U,S,V] = svd(Sigma);
epsilon = 1e-9;
R = U * diag(1./sqrt(diag(S) + epsilon)) * U';
Z = Z * R;

end