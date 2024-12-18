function BiclycleToPoseBenchmark
% Benchmark Bicycle Control Behavior from a set of position

% Create set of starting position and the goal
t=0:pi/10:2*pi;
Starts=[cos(t);sin(t);2*t];
xGoal = [0;0;0];

Perf=[];

% loop from starting positions
for i=1:size(Starts,2)
    xTrue=Starts(:,i);
    k=1;
    while max(abs(dist(xTrue,xGoal)))>.05 && k<10000
        
        % Compute Control
        u=BicycleToPoseControl(xTrue,xGoal);
        
        % Simulate Vehicle motion
        xTrue = SimulateBicycle(xTrue,u);
        
        k=k+1;
    end;
    
    % Store performances
    Perf=[Perf k];
    
end;
% Display mean performances
disp(['Mean goal reaching time : ', num2str(mean(Perf))]);