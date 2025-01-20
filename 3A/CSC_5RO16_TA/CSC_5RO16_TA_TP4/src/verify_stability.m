function ok = verify_stability(x_verif)
    try
    pkg load control
    end

    % Parameters
    mu = 0.5;
    u0 = 0;
    x10 = 0;
    x20 = 0;

    % Weight Matrices
    Q=[0.5, 0; 0, 0.5];
    R=[1];

    % Linearised Jacobians
    A = [u0 * (1 - mu), 1; 1, -u0 * 4 * (1 - mu)];
    B = [mu + (1 - mu) * x10; +mu - 4 * (1 - mu) * x20];

    % Riccati Equations
    [x, l, g] = care(A, B, Q, R);
    K = -g
    % disp(K)


    % Feedback Loop
    Ak = A + B * K;
    % disp(eig(Ak))

    M = [-1, 0; 0, -1] - (Ak);
    % disp(det(M));


    % Limited Lambda
    lambda = -max(eigs(Ak));
    alpha = 0.95 * lambda;

    % Lyapunov Equations
    Al = (Ak + [alpha, 0; 0, alpha])';
    Bl = (Q + K' * R * K);
    assert(all(real(eig(Al)) < 0), "Al is not stable.");
    assert(issymmetric(Bl), "Bl is not symmetric.");
    assert(all(eig(Bl) >= 0), "Bl is not positive semi-definite.");

    P = lyap(Al, Bl);
    assert(issymmetric(P), "P is not symmetric.");


    % Quadratic Problem
    [X, OBJ, INFO, LAMBDA] = qp([0.5; 0.5], -2*P, [], [], [], [-0.8; -0.8], [+0.8; +0.8], -2, K, 2);
    beta = -OBJ;


    % Controller Stable Zone
    test = x_verif' * P * x_verif;
    ok = (test < beta);


endfunction
