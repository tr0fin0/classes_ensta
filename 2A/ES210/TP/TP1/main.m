%%                    Main Code
%%============================================================================

clc
clear all   % ctrl+r:   comment
close all   % ctrl+t: uncomment

RA = 217276;


% 2.2.1
%%============================================================================
PSF1 = load('PSF1.txt');
PSF2 = load('PSF2.txt');

spectre_1 = abs(fft(PSF1,1000) .* conj(fft(PSF1,1000)));
spectre_2 = abs(fft(PSF2,1000) .* conj(fft(PSF2,1000)));

% figure
plot(log(fftshift(spectre_1)), 'LineWidth', 2); hold all
plot(log(fftshift(spectre_2)), 'LineWidth', 2); hold off
grid on; legend('PSF 1', 'PSF 2')
% A signal that is more suited for deconvolution is one that has well-defined peaks or features that can be separated and analyzed using mathematical techniques.
% Ideally, a signal that is suitable for deconvolution would have the following characteristics:
%     High Signal-to-Noise Ratio (SNR): The signal should have a high SNR, which means that the signal is stronger than the noise. This helps to ensure that the peaks or features in the signal are well-defined and easily distinguishable from the noise.
%     Good Resolution: The signal should have good resolution, which means that it should have a high level of detail or information content. This makes it easier to separate the individual peaks or features in the signal and analyze them separately.
%     Minimal Overlap: The signal should have minimal overlap between the individual peaks or features. This makes it easier to separate and deconvolve the individual components of the signal.
%     Linearity: The signal should be linear, which means that the output is directly proportional to the input. This allows for the use of linear deconvolution techniques.
% Overall, a signal that has well-defined peaks or features, high SNR, good resolution, minimal overlap, and linearity is most suited for deconvolution.


% 2.2.2
%%============================================================================

% la borde est assombre a cause de la borne de convolution
% circular est importante que des donnes sont coerentes pour la deconvolution
sigma = 1/255; 
% si le image varie entre 0 et 1 il faut ajouter ce parametre pour que le image ne explose pas

image = double(imread('house_groundtruth.png'));
estimated_nsr = sigma^2/var(image(:));

% 1
image_filtered1 = imfilter(image, PSF1, 'conv', 'circular');
image_filtered1 = image_filtered1 + sigma*randn(size(image));
res1 = deconvwnr(image, PSF1, estimated_nsr);

% la estimation des parametres de regularisation sont importantes et sont un autre probleme qui devrait etre resolu dans le cas commun. dans notre cas ces variables sont dejas donnes pour faciliter la resolution du probleme

% 2
image_filtered2 = imfilter(image, PSF2, 'conv', 'circular');
image_filtered2 = image_filtered2 + sigma*randn(size(image));
res2 = deconvwnr(image, PSF2, estimated_nsr);

% figure
imagesc([image_filtered1, image_filtered2, res1, res2], [0 255])


% 2.3
%%============================================================================

% -- 0 1 1 0 1 1 0 0 0 0 1 0 -- 
%    ___________________ 10 unités de temps
%  la moitie sera des 1
%  il commence et il finisse avec 1
%  s = n / 2

% 1
% donne par la combination entre (n-2 s-2)
% hipotesse entre que il y a une pixel pour second
% la seule diference est la presence entre le ecart typique 
% chaque point corresponde une 
n=10;
s=5;
% generation des codes
code=gene_code(n,s);

% calcul des critères pour chaque code

for u=1:size(code, 2)
    code_teste=code(:,u);
    % normalisation
    PSF = code_teste/sum(code_teste(:));
    spectre_fourrier = abs(fft(PSF, 1000).^2);
    std_fft(u) = std(spectre_fourrier);
    m(u) = min(spectre_fourrier);
end

% on affiche sur un graphe les deux critères
% figure
set(gca, 'FontSize', 15)
plot(std_fft, m, '+', 'LineWidth', 2)
xlabel('écart-type')
ylabel('valeur minimale')


% choisir un point avec le plus grand valeur minimale et le plus petit écart typique
% la solution n'est pas unique il y a plusieurs options à choisir


% conjugaison numérique
% 1/x0' - 1/x0 = 1/f'
% x0 = -2m
% x0'= ?
% 3.1
% f'= 25mm distance focale
% N = 2.8 nombre d'ouverture
% tpx = 5um taille des pixels
% objet entre 1 et 5m de la caméra
% dans la convetion du exercice on considère la distance de x0 comme négatif

% il faut utiliser les unites correctes sur le syst�me internationale donc
% avec distance en metres
x = 1;
x0 = -2;
N = 2.8;
f = 25*1e-3;
D = f/N;
tpx = 5*1e-6;


% ensemble de profondeur
x = - linspace(1,5,1000);

% position de profondeur
xp = f * x0 / (x0 + f);

% en taille du flou de défocalisation
episilon = D * xp * (1/f + 1./x - 1./xp)/tpx;

% plot
% figure
plot(-x, abs(episilon)); hold all
plot(-x, 1); hold off


x_PdC = x(abs(episilon)<=1);


% figure
plot(log(fftshift(spectre_1)), 'LineWidth', 2); hold all
plot(log(fftshift(spectre_2)), 'LineWidth', 2); hold off
grid on; legend('PSF 1', 'PSF 2')

% position de profondeur
fr = 25.1*1e-3;
fg = 25*1e-3;
fb = 24.8*1e-3;
xpr = fr * x0 / (x0 + fr);
xpg = fg * x0 / (x0 + fg);
xpb = fb * x0 / (x0 + fb);

% en taille du flou de défocalisation
episilonr = D * xp * (1/fr + 1./x - 1./xp)/tpx;
episilong = D * xp * (1/fg + 1./x - 1./xp)/tpx;
episilonb = D * xp * (1/fb + 1./x - 1./xp)/tpx;
x_PdCr = x(abs(episilonr)<=1);
x_PdCg = x(abs(episilong)<=1);
x_PdCb = x(abs(episilonb)<=1);
x_PdCtotal = union(x_PdCr, x_PdCg);
x_PdCtotal = union(x_PdCtotal, x_PdCb);
PdCtotal = length(x_PdCtotal)/length(x)

% plot
figure
hold all 
plot(-x, ones(size(episilon)), 'black', 'LineWidth', 1);
plot(-x, abs(episilonr), 'r', 'LineWidth', 2);
plot(-x, abs(episilong), 'g', 'LineWidth', 2); 
plot(-x, abs(episilonb), 'b', 'LineWidth', 2);
plot(-x_PdCr, ones(size(x_PdCr)), 'r', 'LineWidth', 2);
plot(-x_PdCg, ones(size(x_PdCg)), 'g', 'LineWidth', 2);
plot(-x_PdCb, ones(size(x_PdCb)), 'b', 'LineWidth', 2);
hold off
xlabel('distance m')
ylabel('flou en pixel')
title('champs de vision')
grid on; legend('O', 'R', 'G', 'B')

x_PdCr(size(x_PdCr))

lambdar = 0.605;
lambdag = 0.530;
lambdab = 0.465;
R = 15e-3;
f = 25e-3;

fr = fchrom(lambdar, R, f);
fg = fchrom(lambdag, R, f);
fb = fchrom(lambdab, R, f);


episilonr = D * xp * (1/fr + 1./x - 1./xp)/tpx;
episilong = D * xp * (1/fg + 1./x - 1./xp)/tpx;
episilonb = D * xp * (1/fb + 1./x - 1./xp)/tpx;
x_PdCr = x(abs(episilonr)<=1);
x_PdCg = x(abs(episilong)<=1);
x_PdCb = x(abs(episilonb)<=1);
x_PdCtotal = union(x_PdCr, x_PdCg);
x_PdCtotal = union(x_PdCtotal, x_PdCb);
PdCtotal = length(x_PdCtotal)/length(x)

% plot
figure
hold all 
plot(-x, ones(size(episilon)), 'black', 'LineWidth', 1);
plot(-x, abs(episilonr), 'r', 'LineWidth', 2);
plot(-x, abs(episilong), 'g', 'LineWidth', 2); 
plot(-x, abs(episilonb), 'b', 'LineWidth', 2);
plot(-x_PdCr, ones(size(x_PdCr)), 'r', 'LineWidth', 2);
plot(-x_PdCg, ones(size(x_PdCg)), 'g', 'LineWidth', 2);
plot(-x_PdCb, ones(size(x_PdCb)), 'b', 'LineWidth', 2);
hold off
xlabel('distance m')
ylabel('flou en pixel')
title('defocalisation')
grid on; legend('O', 'R', 'G', 'B')
