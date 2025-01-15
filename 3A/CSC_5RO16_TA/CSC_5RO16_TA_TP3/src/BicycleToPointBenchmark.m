function [mean_performance] = BicycleToPointBenchmark( K_rho, K_alpha )
    % Benchmark Bicycle Control Behavior from a set of position

    % Create set of starting position and the goal
    t=0:pi/10:2*pi;
    Starts=[cos(t);sin(t);2*t];
    xGoal = [0;0;0];

    Perf=[];
    plot_graph = 0;

    % loop from starting positions
    for i=1:size(Starts,2)
        xTrue=Starts(:,i);
        XStore = NaN*zeros(3,10000);
        XErrStore = NaN*zeros(5,10000);
        k=1;
        while max(abs(dist(xTrue(1:2),xGoal(1:2))))>.005 && k<10000

            % Compute Control
            u = BicycleToPointControl(xTrue, xGoal, K_rho, K_alpha);

            % Simulate Vehicle motion
            xTrue = SimulateBicycle(xTrue,u);

            k=k+1;

            XErrStore(:,k) = [dist(xTrue,xGoal);u(1);u(2)];
            XStore(:,k) = xTrue;

            % plot every 100 updates
            if(mod(k-2,100)==0  && plot_graph == 1)
                DoBicycleGraphics(xTrue,XStore,XErrStore);
                drawnow;
            end;
        end;

        % Store performances
        Perf=[Perf k];

    end;
    mean_performance = mean(Perf);

end;
% Display mean performances
disp(['Mean goal reaching time : ', num2str(mean_performance)]);
