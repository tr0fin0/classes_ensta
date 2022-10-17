function exercice1(sinusoide)
    % ========================================
    % Exercice 1
    % ========================================


    % considering a signal given by the following equation:
    %   z(t) = a cos(w1 * iTe) + b cos(w2 * iTe) + c cos(w3 * iTe)
    % 
    % where:
    %   w1 = omega_1
    %   w2 = omega_2
    %   w3 = omega_3

    % initial variables from sinusoide data
    N = sinusoide.N;      % number of samples
    Z = sinusoide.Z;      % value  of samples


    gmkv = @gaussMarkov;
    [N_GM,  Z_GM]  = estimation(gmkv, sinusoide);
    
    rlsq = @recursiveLeastSquares;
    [N_RLS, Z_RLS] = estimation(rlsq, sinusoide);

    figure;
    hold on
    plot((1:N), Z)
    plot(N_GM, Z_GM, '-')
    plot(N_RLS, Z_RLS, 'o')
    hold off

    savePlot(...
        'Exercice 1',...
        {'Experiment', 'Gauss-Markov', 'Recursive Squares'},...
        'ma201_pc5_ex1',...
        't', 'z(t)');
end