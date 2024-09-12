function ftot=fchrom(lambda,R,f)
% Lambda : longueur d'onde en um
% R : rayon de courbure du doublet chromatique
% f : focale de la lentille idéale



B=[1.34317774 0.241144399 9.94317969*10^(-1); 1.39757037 0.159201403 1.2686543];
C=[7.04687339*10^(-3) 2.29005*10^(-2) 9.27508256*10;9.95906143*10^(-3) 5.46931752*10^(-2) 1.19248346*100];


n2(1,:)=1-B(1,1)*lambda.^2./(C(1,1)-lambda.^2)-B(1,2)*lambda.^2./(C(1,2)-lambda.^2)-B(1,3)*lambda.^2./(C(1,3)-lambda.^2);
n2(2,:)=1-B(2,1)*lambda.^2./(C(2,1)-lambda.^2)-B(2,2)*lambda.^2./(C(2,2)-lambda.^2)-B(2,3)*lambda.^2./(C(2,3)-lambda.^2);

n=sqrt(n2);

f_add_on=R./(n(2,:)-n(1,:));

ftot=(1./f+1./f_add_on).^(-1);


