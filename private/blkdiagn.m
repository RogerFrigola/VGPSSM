function B = blkdiagn(A,n)
% Repeats matrix A n times to create a block diagonal matrix.

B = zeros(n * size(A));
for i = 1:n
    ii = 1+(i-1)*size(A,1) : i*size(A,1);
    for j = 1:n
        if i ~= j, continue; end
        jj = 1+(j-1)*size(A,2) : j*size(A,2);
        B(ii,jj) = A;
    end
end

end