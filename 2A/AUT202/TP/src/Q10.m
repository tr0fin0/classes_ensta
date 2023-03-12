function eigenValues = Q10(A)
    eigenValues = eig(A);
    for i = 1:size(eigenValues, 1)
        if (real(eigenValues(i)) > 0)
            disp(['warning: system diverge eig ' num2str(eigenValues(i))]);
        end
    end
end