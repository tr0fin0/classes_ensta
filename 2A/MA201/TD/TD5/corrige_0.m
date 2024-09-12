clear all; close all; clc;

% Chargement des donn�es
load('algoEM.mat'); % Contient X et n

% figure;
% plot(X,'o');

M = 10;

lambda_m = 0.5;
mu_1m = 0.1;
sigma_1m = 1;
mu_2m = -0.1;
sigma_2m = 1;

% It�rations de l'algorithme
for m=1:M
    % Etape (E)
    p_1m = zeros(1,n);
    p_2m = zeros(1,n);
    for i=1:n
        x_i = X(i);
        p_1m(i) = fnormale(x_i,mu_1m,sigma_1m)*lambda_m/(fnormale(x_i,mu_1m,sigma_1m)*lambda_m + ...
            fnormale(x_i,mu_2m,sigma_2m)*(1-lambda_m));
        p_2m(i) = 1 - p_1m(i);
    end
    % Etape (M)
    lambda_m = 1/n*sum(p_1m);
    mu_1m = sum(p_1m.*X)/sum(p_1m);
    sigma_1m = sqrt(sum(p_1m.*(X-mu_1m).^2)/sum(p_1m));
    mu_2m = sum(p_2m.*X)/sum(p_2m);
    sigma_2m = sqrt(sum(p_2m.*(X-mu_2m).^2)/sum(p_2m));
    
    % Calcul de la log-vraisemblance
    logVrais = 0;
    for i = 1:n
        logVrais = logVrais + log(lambda_m*fnormale(X(i),mu_2m,sigma_2m)+(1-lambda_m)*fnormale(X(i),mu_2m,sigma_2m));
    end
end

% Affichage des param�tres
lambda_m, mu_1m, sigma_1m, mu_2m, sigma_2m

% Les vrais param�tres (pour comparaison) sont:
% lambda = 0.3;
% mu_1 = 1;
% sigma_1 = 3;
% mu_2 = -4;
% sigma_2 = 0.3;
