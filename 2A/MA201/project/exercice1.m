function exercice1(filtreKalman)
    % ========================================
    % Exercice 1
    % ========================================
    % 
    % 
    % considering a system model given by the following equation:
    %   | x_{k+1} = F x_{k} + B u_{k} + w_{k}
    %   | z_{k}   = H x_{k} + v_{k}
    % 
    % where:
    %   x_{k} : vector of states of system:
    %       x: space        of x coordenate
    %       y: space        of y coordenate
    %      dx: velocity     of x coordenate
    %      dy: velocity     of y coordenate
    %     ddx: acceleration of x coordenate
    %     ddy: acceleration of y coordenate


    mesures = filtreKalman.mesures;        % 
    T       = filtreKalman.Tmesu;          % 


    % Dk: vector of distances
    % ak: vector of angles
    [Dk, ak] = extractData(mesures);

    figure; % plot the data for reference
    plot(T, Dk); formatPlot('Distance', 'Data', 't', 'D_k')
    savePlot('ma201_project_Dk')

    figure; % plot the data for reference
    plot(T, ak); formatPlot('Angle', 'Data', 't', '\alpha_k')
    savePlot('ma201_project_ak')



    % polar coordinates to cartesian coordinates
    Z = [];
    for i = 1 : length(Dk)
        Dx = Dk(i) * cos(ak(i));
        Dy = Dk(i) * sin(ak(i));
        Zk = [Dx; Dy];

        Z = [Z, Zk];
    end

    X = kalmanSetup(Z);


    figure;
    subplot(2,2,1); plot(T, Dk); formatPlot('Distance D_k', {'data', 'kalman'}, 't', 'm')
    subplot(2,2,2); plot(T, Z(1,:), T, X(1,:)); formatPlot('Distance D_x', {'data'; 'kalman'}, 't', 'm')
    subplot(2,2,3); plot(T, Z(2,:), T, X(2,:)); formatPlot('Distance D_y', {'data'; 'kalman'}, 't', 'm')
    subplot(2,2,4); plot(T, ak); formatPlot('Angle \alpha_k', {'data', 'kalman'}, 't', 'degree')
    % savePlot('ma201_project_Dk_filter')




    function [x, y] = extractData(D)
        x = [];
        y = [];

        for i = 1 : length(D)
            V = D(i,:);
            x = [x, V(1)];
            y = [y, V(2)];
        end
    end

end