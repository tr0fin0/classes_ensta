function DoBicycleGraphicsPath(x,XStore,Path)

figure(1);
hold off;

DrawRobot(x,'b',Path(:,end));
plot(Path(1,:),Path(2,:),'g');
plot(XStore(1,:),XStore(2,:));
plot(Path(1,end),Path(2,end),'+r');



%-------- Drawing Vehicle -----%
function DrawRobot(Xr,col,xGoal);

p=0.02; % percentage of axes size 
a=axis;
l1=(a(2)-a(1))*p;
l2=(a(4)-a(3))*p;
P=[-1 1 0 -1; -1 -1 3 -1];%basic triangle
theta = xGoal(3)-pi/2;%rotate to point along x axis (theta = 0)
c=cos(theta);
s=sin(theta);
P=[c -s; s c]*P; %rotate by theta
P(1,:)=P(1,:)*l1+xGoal(1); %scale and shift to x
P(2,:)=P(2,:)*l2+xGoal(2);
plot(P(1,:),P(2,:),'r','LineWidth',0.1);
hold on;

P=[-1 1 0 -1; -1 -1 3 -1];%basic triangle
theta = Xr(3)-pi/2;%rotate to point along x axis (theta = 0)
c=cos(theta);
s=sin(theta);
P=[c -s; s c]*P; %rotate by theta
P(1,:)=P(1,:)*l1+Xr(1); %scale and shift to x
P(2,:)=P(2,:)*l2+Xr(2);
H = plot(P(1,:),P(2,:),col,'LineWidth',0.1);% draw

axis([-2 5 -1 4]); axis equal;
plot(Xr(1),Xr(2),sprintf('%s+',col));