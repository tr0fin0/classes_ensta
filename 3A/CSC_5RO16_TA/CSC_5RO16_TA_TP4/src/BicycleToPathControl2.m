function [ u ] = BicycleToPathControl2( xTrue, Path )
%Computes a control to follow a path for bicycle
%   xTrue is the robot current pose : [ x y theta ]'
%   Path is set of points defining the path : [ x1 x2 ...
%                                               y1 y2 ...]
%   u is the control : [v phi]'

persistent goalWaypointId xGoal;

%set first waypoint and goal on the path when starting trajectory
if xTrue==[0;0;0]
    goalWaypointId=1;
    xGoal=Path(:,1);
end

rho=0.3;
dt=.01; % integration time (same than simulation)



%% define goals and size of window
window_size=5;   %1 : anticipe, 5 : anticipe bien, controle plus souple, 
%20 : coupe un peu, controle très souple, 100 : coupe un peu, 1000 : triche !
vmax=2.0;
dmax = vmax * dt; %distance max for one step

list_points=[];
xtemp=xTrue; %start from our position


%%TODO : remplir la liste de points en suivant l'explication dans le sujet
while size(list_points,2) < window_size
%% 
end


anticipation = window_size;  %which future points is the objective
%% Then perform P control
    Krho=10;
    Kalpha=5;
 
    error=list_points(:,anticipation)-xTrue;
    goalDist=norm(error(1:2));
    AngleToGoal = AngleWrap(atan2(error(2),error(1))-xTrue(3));

    u(1) = Krho * goalDist/(window_size*10); %vitesse bornée
    u(2) = Kalpha*AngleToGoal;

end

