function UnicycleToPose
% Display Unicycle Control Behavior from random position

% Goal and random starting position
xGoal = [0;0;0];
a=rand()*2*pi;
xTrue = [2*cos(a);2*sin(a);rand()*2*pi];

%Storage for position and errors
XStore = NaN*zeros(3,10000);
XErrStore = NaN*zeros(3,10000);
k=1;

% loop until goal reached or max time
while max(abs(dist(xTrue,xGoal)))>.06 && k<10000
    
    % Compute Control
    u=UnicycleToPoseControl(xTrue,xGoal);
    
    % Simulate Vehicle motion
    [xTrue,u] = SimulateUnicycle(xTrue,u);
    
    k=k+1;
    %store results:
    XErrStore(:,k) = dist(xTrue,xGoal);
    XStore(:,k) = xTrue;
    
    % plot every 100 updates
    if(mod(k-2,100)==0)
        DoUnicycleGraphics(xTrue,XStore,XErrStore);
        drawnow;
    end;

end;

% Draw final position and error curves
DoUnicycleGraphics(xTrue,XStore,XErrStore);
drawnow;
figure(2);
subplot(3,1,1);plot(XErrStore(1,:));
title('Error');ylabel('x');
subplot(3,1,2);plot(XErrStore(2,:));
ylabel('y');
subplot(3,1,3);plot(XErrStore(3,:)*180/pi);
ylabel('\theta');xlabel('time');



