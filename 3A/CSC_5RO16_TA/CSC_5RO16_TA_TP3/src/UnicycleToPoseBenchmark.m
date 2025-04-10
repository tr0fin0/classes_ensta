function [ mean_performance ] = UniclycleToPoseBenchmark( alpha_maximum, K_rho, K_alpha, K_beta )
    % Benchmark Unicycle Control Behavior from a set of position

    % Create set of starting position and the goal
    t=0:pi/10:2*pi;
    Starts=[cos(t);sin(t);2*t];
    xGoal = [0;0;0];
    XStore = NaN*zeros(3,10000);
    XErrStore = NaN*zeros(3,10000);

    plot_graph = 0;

    Perf=[];

    % loop from starting positions
    for i=1:size(Starts,2)
        xTrue=Starts(:,i);
        k=1;
        while max(abs(dist(xTrue,xGoal)))>.005 && k<10000

            % Compute Control
            [ u ] = UnicycleToPoseControl(xTrue, xGoal, alpha_maximum, K_rho, K_alpha, K_beta);

            % Simulate Vehicle motion
            xTrue = SimulateUnicycle(xTrue,u);

            k=k+1;
            max(abs(dist(xTrue,xGoal)));

            XErrStore(:,k) = dist(xTrue,xGoal);
            XStore(:,k) = xTrue;

            if(mod(k-2,100)==0 && plot_graph == 1)
                DoUnicycleGraphics(xTrue,XStore,XErrStore);
                drawnow;
            end;
        end;
        Perf=[Perf k];
    end;

    mean_performance = mean(Perf);
end;
% Display mean performances
disp([num2str(alpha_maximum), ',', num2str(K_rho), ',', num2str(K_alpha), ',', num2str(K_beta), ',', num2str(mean(Perf))]);

