function Sigma2 = stacksigma(Sigma, numStates)

n = size(Sigma,1) / numStates;
Sigma2 = nan(n, n, numStates);
for i = 1:numStates
    ii = (i-1)*n+1:i*n;
    Sigma2(:,:,i) = Sigma(ii,ii);
end

end