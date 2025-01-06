function BicycleToPath2
% Display Bicycle Control Behavior while following a path

% Path and starting position
Path = [.1,0,0; 4,0,0; 4,4,0; 3.5,1,0; 0,4,0;0,1,-1.57]';
xTrue = [0;0;0];

%Storage for position and errors
XStore = NaN*zeros(3,20000);
XErrStore = zeros(3,20000);
k=1;

% loop until goal reached or max time
while max(abs(dist(xTrue(1:2,1),Path(1:2,end))))>.005 && k<200000 %20000
    % Compute Control
    u=BicycleToPathControl2(xTrue,Path);
    
    % Simulate Vehicle motion
    [xTrue,u] = SimulateBicycle2(xTrue,u);
    
    k=k+1;
    %store results:
    error=abs(p_poly_dist(xTrue(1),xTrue(2),Path(1,:),Path(2,:)));
    XErrStore(:,k) = [error;u'];
    XStore(:,k) = xTrue;
    
    % plot every 100 updates
    if(mod(k-2,100)==0)
        DoBicycleGraphicsPath(xTrue,XStore,Path);
        drawnow;
    end;
    
end;

% Draw final position and error curves
DoBicycleGraphicsPath(xTrue,XStore,Path);
drawnow;
figure(2);
subplot(3,1,1);plot(XErrStore(1,:));
title('Error');ylabel('dist');
xlabel('time');
subplot(3,1,2);plot(XErrStore(2,:));
title('Controls');ylabel('v');
subplot(3,1,3);plot(XErrStore(3,:)*180/pi);
ylabel('\phi');xlabel('time');

disp(['Total error : ',num2str(sum(XErrStore(1,:)))]);



