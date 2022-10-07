function theta = gaussMarkov(input)
    % ========================================
    % Gauss-Markov Method
    % ========================================


    % initial variables
    N       = input.N;
    Te      = input.Te;
    Z       = input.Z;
    om1     = input.om1;
    om2     = input.om2;
    om3     = input.om3;
    sigmab  = input.sigmab;

    % initial matrixes
    Z_1N = Z;
    H_1N = [cos(om1 * (1:N)' * Te), cos(om2 * (1:N)' * Te), cos(om3 * (1:N)' * Te)];
    m_1N = zeros(N, 1);
    R_1N = sigmab * eye(N);


    % theorical expression
    % theta = inv(inv(R_T) + H_1N' * inv(R_1N) * H_1N) * (inv(R_T) + Z_1N - m_1N));

    % optimized algorithm
    % here the R_T matrix was consider as zero because... TODO search answer
    % theta = (H_1N' / R_1N * H_1N) \ (H_1N' * (Z_1N - m_1N));        % slide algorithm
    theta = (H_1N' / R_1N * H_1N) \ (H_1N' / R_1N) * (Z_1N - m_1N); % teacher algoritm

end