clear all; close all; clc;
load 'filter.mat'
% calcul des coefficients
for j = 2:7
P_k = 10*eye(j);
parx=ones(j,1);
for n = 1:N-7
    hx = X(n+j-1:-1:n)';
    rx = hx*P_k*hx' + sigmab;
	kx = P_k*hx'/rx;
	P_k = P_k-kx*hx*P_k;
	parx = parx + kx*(Z(n+j)-hx*parx);
end
H=[-4 5 -3 10 -20]';%(-0.5+1*rand(5,1));
parx
% calcul de akaike
% Le bruit �tant de moyenne nulle, la log vraisemblance correspond �
% l'erreur quadratique 
errquad=0.;
for n =1:N-7
    hx = X(n+j-1:-1:n)';
    zmes(n+j) = hx*parx;
    diffcar = (Z(n+j)-hx*parx)*(Z(n+j)-hx*parx);%sigmab;
    diffplot(n) = sqrt(diffcar);
    errquad = errquad+diffcar;
end
figure
plot(Z,'b')
hold on
plot(zmes,'r')
Akaike(j)=2*j-2.*errquad;
Akaikecorr(j)=Akaike(j)+2*j*(j+1)/(N-j-1);
end
figure
plot(Akaike(2:7))
hold on
plot(Akaikecorr(2:7))
% selection de l'orde
% on constate que les valeurs de crit�re deviennent tr�s proches 
jj=max(Akaike)
find(Akaike==jj)
jj=max(Akaikecorr)
find(Akaikecorr==jj) 