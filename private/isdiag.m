function answer = isdiag (A)

    % degenerate shape:
    if ( isempty(A) )
        answer = false;
        return
    elseif isscalar(A)
        answer = true;
        return
    end

    siz = size(A);
    ind_diag = sub2ind(siz, 1:siz(1), 1:siz(2))';  % index of diagonal elements.
    ind_nz = find(A);  % index of non-zero elements.
    %ind_diag, ind_nz, all(ismember(ind_nz, ind_diag))  % DEBUG
    answer = isempty(ind_nz) || all(ismember(ind_nz, ind_diag));
end