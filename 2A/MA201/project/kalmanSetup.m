function X = kalmanSetup(Z, Dk, Ak)
    % ========================================
    % Kalman Filter Setup
    % ========================================


    n = 6;  % number of states
    m = 2;  % number of outputs

    % Vector Initial conditions (6 x 1)
    x   = [ 0; 0; 0; 0; 0; 0 ];

    % Constants
    T   = 0.1;      % sample time
    U   = 1/2*T^2;  % constant
    b   = 1;        % beta
    g   = 9.81;     % gravity

    % Matrix of Transition      (6 x 6)
    F   = [
        1 0 T 0 U 0;
        0 1 0 T 0 U;
        0 0 1 0 T 0;
        0 0 0 1 0 T;
        0 0 0 0 b 0;
        0 0 0 0 0 b-g;
    ];

    % Matrix of Control         (6 x 2)
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

    u   = wgn(m, length(Z), 0;  % Generate white Gaussian noise samples

    %   = (mean + variance * normal distribution values)
    Q   = (0 + 1 * randn(n));   % Matrix of Covariance Noise of wk
    R   = (0 + 1 * randn(m));   % Matrix of Covariance Noise of vk


    X = kalmanFilterSimple(Z, x, F, B, H, Q, R, u);

end