function exercice1(filtrePart)
    % ========================================
    % Exercice 1
    % ========================================


    % considering a system model given by the following equation:
    %   | x_{k+1} = F x_{k} + w_{k}
    %   | z_{k}   = H x_{k} + v_{k}
    % 
    % where:
    %   x_{k} : vector of states of system:
    %       x: space    of x coordenate
    %      dx: velocity of x coordenate
    %       y: space    of y coordenate
    %      dy: velocity of y coordenate
    %   z_{k} : vector of mesures
    %   w_{k} : noise of model  given by: N(0, W)
    %   v_{k} : noise of mesure given by: N(0, V)
    %       N(x,y) : Normal Distribution



    N = filtrePart.N; % 
    X = filtrePart.X; % 
    T = filtrePart.T; % 

    % getting real values for comparison
    [x, y, dx, dy] = extractData(X, N);



    [Xk, tk] = kalmanCalculation(filtrePart);
    [xk, yk, dxk, dyk] = extractData(Xk, N);

    t = 1 : 1 : N;
    figure;
    subplot(2,2,1); plot(t,  x, t,  xk); formatPlot( 'x', {'data', 'kalman'}, 't',  'x(t)')
    subplot(2,2,2); plot(t, dx, t, dxk); formatPlot('dx', {'data', 'kalman'}, 't', 'dx(t)')
    subplot(2,2,3); plot(t,  y, t,  yk); formatPlot( 'y', {'data', 'kalman'}, 't',  'y(t)')
    subplot(2,2,4); plot(t, dy, t, dyk); formatPlot('dy', {'data', 'kalman'}, 't', 'dy(t)')
    savePlot('ma201_pc6_ex1')

    figure;
    plot(t, tk); formatPlot('trace(P_k)', 'Kalman Filter', 't', 'tr(P_k)')
    savePlot('ma201_pc6_ex2_trace')

    figure;
    plot(t, tk); formatPlot('trace(P_k)', 'Kalman Filter', 't', 'tr(P_k)')
    savePlot('ma201_pc6_ex2_erreur')



    function [x, y, dx, dy] = extractData(X, N)
        x = []; dx = [];
        y = []; dy = [];

        for i = 1 : N
            V = X(:,i);
            x = [x, V(1)]; dx = [dx, V(2)];
            y = [y, V(3)]; dy = [dy, V(4)];
        end
    end



end