function Y = P321(X)
% Shorthand for common permuting dimensions 1 and 3 in a 3d array.

Y = permute(X, [3 2 1]); 

end