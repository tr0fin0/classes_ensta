function DoUnicycleGraphics(x,XStore,XErrStore)

figure(1);
hold off;
DrawRobot(x,'b');
plot(XStore(1,:),XStore(2,:));
plot(0,0,'+r');



%-------- Drawing Vehicle -----%
function DrawRobot(Xr,col);

p=0.02; % percentage of axes size 
a=axis;
l1=(a(2)-a(1))*p;
l2=(a(4)-a(3))*p;

P=[-3 -3 -3 -2 -2 2 2 3 3 3  3 2  2 -2 -2; 
   -1  1  0  0  2 2 0 0 1 -1 0 0 -2 -2 0]/2;%robot shape

theta = -pi/2;%rotate to point along x axis (theta = 0)
c=cos(theta);
s=sin(theta);
P=[c -s; s c]*P; %rotate by theta
P(1,:)=P(1,:)*l1; %scale and shift to x
P(2,:)=P(2,:)*l2;
plot(P(1,:),P(2,:),'r','LineWidth',0.1);
hold on;
P=[-3 -3 -3 -2 -2 2 2 3 3 3  3 2  2 -2 -2; 
   -1  1  0  0  2 2 0 0 1 -1 0 0 -2 -2 0]/2;%robot shape
theta = Xr(3)-pi/2;%rotate to point along x axis (theta = 0)
c=cos(theta);
s=sin(theta);
P=[c -s; s c]*P; %rotate by theta
P(1,:)=P(1,:)*l1+Xr(1); %scale and shift to x
P(2,:)=P(2,:)*l2+Xr(2);
H = plot(P(1,:),P(2,:),col,'LineWidth',0.1);% draw

axis([-2 2 -2 2]); axis equal;
plot(Xr(1),Xr(2),sprintf('%s+',col));