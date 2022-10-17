function particulesCalculation(input)
    % ========================================
    % Particules Filter Calculation
    % ========================================
    % x : vector of values
    % t : vector of trace


    % input variables
    N       = input.N; % 
    T       = input.T; % dicreatization time
    X       = input.X; % real values
    Z       = input.Z; % 
    sigma_p = input.sigma_p; % 
    sigma_v = input.sigma_v; % 
    sigma_z = input.sigma_z; % 

    % initial variables
    sp2 = sigma_p ^ 2;
    sv2 = sigma_v ^ 2;
    sz2 = sigma_z ^ 2;


    F = [
        1 T 0 0;
        0 1 0 0;
        0 0 1 T;
        0 0 0 1;
    ];

    H = [
        1 0 0 0;
        0 0 1 0;
    ];

    W = T * [
        sp2 0   0   0;
        0   sv2 0   0;
        0   0   sp2 0;
        0   0   0   sv2;
    ];

    % TODO check matrix dimensions and kalman algorithm needs
    V = sz2 * eye(2);
    % F, H, W and V are constant during interations, particular case


    % inicialization
    randRange(50, 100)
    % [x, t]  = particuleFilter(N, Z, F, H, W, V);


    function r = randRange(a, b)
        r = round((b-a).*rand(1) + a);
    end
end