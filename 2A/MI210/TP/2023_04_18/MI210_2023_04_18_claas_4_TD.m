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

%   according to chatGPT:
%       covariance is a statistical measure that describes how two variables
%       are related to each other. More specifically, it measures the degree
%       to which two variables vary together. If the two variables tend to
%       increase or decrease together, then the covariance is positive. 

%       If one variable tends to increase as the other decreases, then the
%       covariance is negative. If there is no relationship between the two
%       variables, then the covariance is zero.

%       Mathematically, the covariance between two random variables X and Y
%       with means μX and μY, respectively, is given by:

%           Cov(X, Y) = E[(X - μX) (Y - μY)]

%       where E is the expected value operator.


%   according to chatGPT:
%       A decorrelating matrix is a matrix that transforms a set of variables
%       into a new set of variables that are uncorrelated with each other.

%       It is often used in signal processing and data analysis to reduce the
%       correlation between variables and to simplify the analysis of data.

%       The decorrelating matrix is calculated from the covariance matrix of 
%       the original variables, and its columns form a new basis that is
%       orthonormal, i.e., the columns are perpendicular to each other and have
%       unit length.

%       When the original variables are multiplied by the decorrelating matrix,
%       the resulting variables have zero correlation with each other, and their
%       variances are maximally spread out along the new axes.

%       The decorrelating matrix is also known as the whitening matrix because
%       it transforms the data into a white noise signal that has equal power at
%       all frequencies.

%   in this case the mean is zero and therefore it is not considered
cov_x = cov(x');
%   MATLAB considers observations on each row and variables on each collumn
%   but in the dataset it is the opposity, so transpose is needed

%   compute z = WZ*x, and its covariance
z = WZ * x; 
cov_z = cov(z');

%   visualization
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
figure('Name', 'images without de-noising')
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


%%  Exercise 4: de-noising filter
%   ===========================================================================

%   in order to continue using the white filter a denoise filter can be
%   applied before appling the white filter.

%   examples of possible approches:
%       smoothing;
%       wavelet de-noising;
%       Principal Component Analysis, PCA;
%       Independent Component Analysis, ICA;

%   array of noises
eta = [0.0 0.3 1.4]*std(x(:));

figure('Name', 'de-noising')

%   analysis over different noise levels
for i = 1:length(eta)

    % where <...> is consider as average of values
    %   <(x + eta)(x + eta)'>
    %   <x x'> + 2<x eta'> + <eta eta'>
    %   cov(x) + sigma2 * I
    
    Wn = (cov_x + eta(i)^2 * eye(length(cov_x))) \ cov_x;
    % Wn = (<(x + eta)(x + eta)'>)^{-1} <x (x + eta)>
    %   is prefereable to use the '\' operator instead of the inverse
    %   operator for numerical reasons

    % where the order of the multiplication with influence inthe result of
    % the operation from right to left, first is needed to compute the 
    % de-noising matrix to after compute the original white filter matrix
    Wcombined = WZ * Wn;

    % extracting a part of the matrix
    wcombined = reshape(Wcombined(:, 78), 12, 12);
    
    subplot(1, 3, i)
    imagesc(wcombined); axis square; axis on;
    title(['eta(', num2str(i), ')'])
end

%   where we are studying how different de-noising filters can remove noise
%   added to the original image and their visual representation.

%   de-noising will be more efficient if the value of de-noise and noise match
%   therefore the filter will be able to correctly identify and correct.



%%  Exercise 5:
%   ===========================================================================

%   defintion noise
eta = 0.7*std(x(:));

%   computing de-noising matrix
Wn = (cov_x + eta^2 * eye(length(cov_x))) \ cov_x;

%   combinning the filters
Wcombined = WZ*Wn;

%   extracting a part of the matrix
w = reshape(Wcombined(78, :), 12, 12);

%   convolution of the noisyless image with a combined filter
Z = conv2(X, w, 'same');

%   adding noise to the original image
noise = eta * randn(size(X));
Xn = X + noise;

%   convolution of the noisy image with a filter
Zn = conv2(Xn, w, 'same');

figure('Name', 'images with de-noising')
subplot(2, 2, 1)
imagesc(X);
title('orignal image')

subplot(2, 2, 2)
imagesc(Z);
title('convolution of original image')

subplot(2, 2, 3)
imagesc(Xn);
title('orignal image with noise')

subplot(2, 2, 4)
imagesc(Zn);
title('convolution of original image with noise')
colormap('gray')

%%  plot the ICA, Independent Component Analysis, filters
%   ===========================================================================

%   according to chatGPT:
%       In signal processing, Independent Component Analysis (ICA) is a technique 
%       used to separate a multivariate signal into independent, non-Gaussian 
%       components. One application of ICA is to separate a mixed signal into its 
%       original source signals.

%       The ICA filter is a mathematical operation used to perform this separation. 
%       It is essentially a matrix that is applied to the mixed signal to produce 
%       a set of output signals that are as independent as possible. The ICA filter 
%       is designed to maximize the non-Gaussianity of the output signals, which is 
%       a measure of their statistical independence.

%       The ICA filter is typically computed using an optimization algorithm that 
%       seeks to find the set of filter coefficients that maximizes the non-Gaussianity 
%       of the output signals. Once the filter is computed, it can be applied to the 
%       mixed signal to produce the set of independent component signals. The ICA filter 
%       is often used in applications such as blind source separation, where the goal 
%       is to recover the original signals from a set of mixed signals without any prior 
%       knowledge of the mixing process.

figure('Name', 'ICA filters')
for i = 1:25
    subplot(5, 5, i)
    imagesc(reshape(WI(i, :), 12, 12)); colormap('gray'); axis off; axis square
    title(['ICA(', num2str(i), ')'])
end


%%  Exercise 6: plot a histogram of Z and Z_ICA
%   ===========================================================================

hist_values = -100:1:100; % grid for histogram

%   Z values decorrelated
Z_DEC = WZ * x; % covolution with filter

%   Z values of ICA analysis
Z_ICA = WI * x; % covolution with filter

%   computing histogram
hist_DEC = hist(Z_DEC(:), hist_values);
hist_ICA = hist(Z_ICA(:), hist_values);

figure('Name', 'ICA histograms')
bar(hist_values, [hist_DEC; hist_ICA]'); hold on
plot(hist_values, hist_DEC, 'b'); hold on
plot(hist_values, hist_ICA, 'r'); hold on
set(gca, 'YScale', 'log')
xlim([-100 100])
grid on
axis square
legend('decorrelated', 'independent component')
title(['ICA histogram'])

%   note that the Z_ICA algorithm is heavy tailed, which means acoording to chatGPT:
%       In probability theory and statistics, a heavy-tailed distribution is a 
%       probability distribution that has a larger proportion of observations in 
%       its tail than would be expected if the distribution were normal or Gaussian. 

%       In other words, a heavy-tailed distribution has more extreme values or 
%       outliers than a normal distribution.

%       The tail of a distribution refers to the region of the distribution that 
%       contains the largest values or extreme observations. A heavy-tailed 
%       distribution has a slower decay rate in the tail than a normal distribution, 
%       which means that extreme values are more likely to occur. This can result 
%       in a higher frequency of extreme events, such as stock market crashes or 
%       natural disasters.

%       Some examples of heavy-tailed distributions include the Cauchy distribution, 
%       the Student's t-distribution, and the power-law distribution. Heavy-tailed 
%       distributions are often used to model complex systems in which extreme events 
%       are more common, such as in finance, climate science, and network analysis.


%   also note that Z_ICA is sparsely code, in other words as defined by chatGPT:
%       In neuroscience and machine learning, sparse coding is a method of representing 
%       data in which each item is represented by a small number of active neurons or 
%       features. In other words, a sparse code is a way of expressing information using 
%       only a subset of the available features or variables.

%       The idea behind sparse coding is that natural signals, such as images or sounds, 
%       are often highly redundant and can be efficiently represented by a small number 
%       of features. By selectively activating only a few neurons or features, a sparse 
%       code can reduce the dimensionality of the data and simplify the processing 
%       required to analyze it.

%       Sparse coding has been observed in the brain, where neurons in the visual cortex 
%       have been shown to respond selectively to specific features of visual stimuli. 
%       In machine learning, sparse coding is often used as a feature selection or 
%       dimensionality reduction technique, where the goal is to find a compact 
%       representation of the data that preserves its essential structure.


%   looking to the graph we can see, with the help of chatGPT, that:
%       Alternatively, you can plot a histogram of the amplitude values of all
%       the ICs together to get an overall view of the data distribution.

%       The histogram bars show how many ICs have amplitudes in a particular
%       range of values. The overall shape of the histogram can provide insights
%       into the structure of the data, such as the presence of outliers or
%       whether the data is normally distributed. The histogram can also help
%       identify the most prominent ICs by looking at the peaks in the
%       distribution.


%%  plot conditional histogram of  
%   ===========================================================================

%   now independance of the variables is going to be evaluated to determine
%   the behavior of the code 

%   computing p(z1, z2)
pz1z2 = hist3(Z_ICA([9, 10], :)', {hist_values' hist_values'});

%   computing p(z2)
pz2 = sum(pz1z2, 1);

%   computing p(z1 | z2), conditional probability
pz1_z2 = pz1z2 ./ (pz2 + 1e-5);
%   note that 1e-5 as added to avoid infinite numbers by zero division
%   during the calculus over the array.

figure('Name', 'conditional histogram')
imagesc(hist_values, hist_values, log(pz1_z2+0.01))
set(gca, 'Xlim', [-10 10], 'Ylim', [-10 10])
grid on; 
axis square;
colormap('gray')

%   note that best linear filters are not independet as shown in the plot
%   there is a bold like shape on the image, that means that they are not 
%   independent.