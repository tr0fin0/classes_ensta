function X = kalmanSetup(input)
    % ========================================
    % Kalman Filter Setup
    % ========================================


    n = 6;
    m = 2;

    % Vector Initial conditions (6 x 1)
    x   = [ 0; 0; 0; 0; 0; 0 ];

    % Matrix of Transition      (6 x 6)
    T   = 0.1;
    U   = 1/2*T^2;
    b   = 1;
    
    F   = [
        1 0 T 0 U 0;
        0 1 0 T 0 U;
        0 0 1 0 T 0;
        0 0 0 1 0 T;
        0 0 0 0 b 0;
        0 0 0 0 0 b;
    ];

    % Matrix of Control         (6 x 2) ?????
    B   = [
        0 0;
        0 0;
        0 0;
        0 0;
        1 0;
        0 1;
    ];

    % Matrix of Observation     (2 x 6)
    H   = [
        1 0 0 0 0 0;
        0 1 0 0 0 0;
    ];

    u   = zeros(2, 1);
    Q   = zeros(n, n);  % Matrix of Covariance Noise of wk
    R   = zeros(m, m);  % Matrix of Covariance Noise of vk

    X = kalmanFilterSimple(input, x, F, B, H, Q, R, u)

end