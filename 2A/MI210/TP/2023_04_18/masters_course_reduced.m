%%  2023_04_18 TP 3
%   MI210 - Neurocomputational Models
%   ===========================================================================

clear
clc
close all


%%  Exercise 1: entropy of 2 neurons, with varying covariance matrices
%   ===========================================================================

%   create an array of matrices
sigmas = {[1 0.00; 0.00 1] [1 0.50; 0.50 1] [1 0.99; 0.99 1]};

%   calculation of entropy H
for i=1:length(sigmas)
    % using the natural logarithme
    H = 1/2 * log( 2*pi*exp(1) * det(sigmas{i}) )
end

%   as expected entropy is higher on more random sets of data and lower on
%   less random sets of data.

%   that makes sense, if the data is predictible it means that it does not 
%   give much usefull information.

%   note that the entropy can be negative, which indicates a predictible
%   set of data.


%%  Exercise 2: 
%   ===========================================================================

load('data.mat')

% WZ: decorrelating filters
% WI: ICA filters, need for Exercice 4

% x: 16000 image patches of size 12 x 12 = 144
% X: 1 full image 

%   compute the covariance of x
%   cov_x = x * x' / size(x,2) - mean(x)
%   another way of computing the covariance matrix is defined above.

%   in this case the mean is zero and therefore it is not considered
cov_x = cov(x');
%   MATLAB considers observations on each row and variables on each collumn
%   but in the dataset it is the opposity, so transpose is needed

%   compute z = WZ*x, and its covariance
z = WZ * x; 
cov_z = cov(z');

figure('Name', 'covariance')

subplot(1, 2, 1)
imagesc(cov_x(1:12, 1:12)); 
grid on
axis square
colormap('gray')
title('covariance of stimulus')

subplot(1, 2, 2)
imagesc(cov_z(1:12, 1:12)); 
grid on
axis square
colormap('gray')
title('covariance of Z')
%   white means high correlation and black means low correlation. values in
%   between should follow the same scale.


%%  Exercise 3: 
%   ===========================================================================

%   reshape image with whitinning filter into a patch of data
w = reshape(WZ(78, :), 12, 12);

%   convolution of the image with a filter
Z = conv2(X, w, 'same');

%   showing original image for benchmark
figure('Name', 'images without denoising')
subplot(2, 2, 1)
imagesc(X);
title('original image')

subplot(2, 2, 2)
imagesc(Z, [-1 1]);
title('convolution of original image')

%   computing stardard deviation
eta = 0.5*std(x(:));

%   adding noise to the original image
noise = eta * randn(size(X));
Xn = X + noise;

%   convolution of the noisy image with a filter
Zn = conv2(Xn, w, 'same');

subplot(2, 2, 3)
imagesc(Xn);
title('original image with noise')

subplot(2, 2, 4)
imagesc(Zn, [-1 1]);
title('convolution of original image with noise')
colormap('gray') % comment for colorfull image
%   amplify the noise and suppress the signal of the image
%   even a small amount of the stimulus will destroy the image

%   this filter is not robust agaist noise in this proposition

%   according to chatGPT:
%       A white filter is a linear filter that passes all frequencies 
%       equally, and does not alter the amplitude or phase of any 
%       frequency component of the signal. In other words, it has a 
%       flat frequency response across all frequencies.

%       In signal processing, a white filter is often used to filter 
%       noise from a signal while preserving the signal's frequency 
%       content. It can also be used to equalize the frequency response 
%       of a system, for example in audio systems to compensate for 
%       uneven speaker response.


    % Wn = 

    % Wcombined = 

    wcombined = reshape(Wcombined(:, 78), 12, 12);
    
    subplot(1, 3, i)
    imagesc(wcombined); axis square; axis off;
end


%% %% Exercise 5:
eta = 0.7*std(x(:));
% Wn =

% Wcombined = WZ*Wn;

w = reshape(Wcombined(78, :), 12, 12);

 Z = conv2(X, w, 'same');

% noisy version of stimulus
% Xn = 

% convolve w with Xn to get response
% Zn = 

figure('Name', 'images')
subplot(2, 2, 1)
imagesc(X);
subplot(2, 2, 2)
imagesc(Z);
subplot(2, 2, 3)
imagesc(Xn);
subplot(2, 2, 4)
imagesc(Zn);
colormap('gray')

%% plot the ICA filters
figure('Name', 'ICA filters')
for i = 1:25
    subplot(5, 5, i)
    imagesc(reshape(WI(i, :), 12, 12)); colormap('gray'); axis off; axis square
end

%% Exercise 6: plot a histogram of Z and ZICA

grd = -100:1:100; % grid for histogram

% Z = 
% ZICA = 

% pz = 
% pr = 

figure('Name', 'responses')
semilogy(grd, pz, 'k'); hold on
semilogy(grd, pr, 'r'); hold on
legend('decorrelated', 'independent component')

%% plot conditional histogram of  

pz1z2 = hist3(ZICA([9, 10], :)', {grd' grd'});

% TODO compute p(z_2)
% pz2 = 

% TODO compute p(z_1|z_2)
% pz1_z2 = 

figure('Name', 'conditional histogram')
imagesc(grd, grd, log(pz1_z2+0.01))
set(gca, 'Xlim', [-10 10], 'Ylim', [-10 10])
colormap('gray')