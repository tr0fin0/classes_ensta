function theta = gaussMarkov(input)


    N       = input.N;
    Te      = input.Te;
    Z       = input.Z;
    om1     = input.om1;
    om2     = input.om2;
    om3     = input.om3;
    sigmab  = input.sigmab;


    m_1N = zeros(N, 1);

    R_1N = sigmab * eye(N);


    H_1N = [cos(om1 * (1:N)' * Te), cos(om2 * (1:N)' * Te), cos(om3 * (1:N)' * Te)];
    % H_1N = [cos(om1 * (1:N))' * Te cos(om2 * (1:N))' * Te cos(om3 * (1:N))' * Te];

    theta = (H_1N' / R_1N * H_1N) \ (H_1N' / R_1N) * (Z_1N - m_1N);
    % theta = inv(H_1N' / R_1N * H_1N) * (H_1N' / R_1N) * (Z_1N - m_1N);
    return theta
end