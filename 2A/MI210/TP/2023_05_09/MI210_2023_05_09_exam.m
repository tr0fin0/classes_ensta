%%  2023_05_09 examen
%   MI210 - Neurocomputational Models
%   ===========================================================================

clear
clc
close all

whos -file data_exam.mat    % sees  data
load('data_exam.mat');


%%  introduction 
%   ===========================================================================

%   size of the stimulus
[nx, ~, N] = size(X);
%   nx: 30      size of image patch in one direction
%   N:  2000    number of trials

%   plot one example stimulus
figure('Name', 'stimulus: 1')
imagesc(X(:, :, 1))
axis square
colormap('gray')

%   plot one example stimulus
figure('Name', 'histogram of spike counts: 1')
hist(r, 100)                          
xlabel('recorded spike counts')
ylabel('number')


%%  exercise 1: compute spike triggered average
%   ===========================================================================

%   reshape stimulus: number of pixels (nx^2) x number of trials (N)
Xvec = reshape(X, nx^2, N);

%   remove mean from stimulus 
Xtilde = zeros(size(Xvec)); rtilde = zeros(size(r));      % remove this line
Xtilde = Xvec - mean(mean(Xvec)); % ex 1(a) remove mean from stimulus                
rtilde = r - mean(r);       % ex 1(a) remove mean from firing rate


cov_x = cov(Xvec');
figure('Name', 'covariance')
imagesc(cov_x); 
grid on
axis square
colormap('gray')
title('covariance of stimulus')

% ex 1(b) can you show that the stimulus is 'white' (see exercise)?
%   we can see that the stimulus is white because it only has covariance in
%   the main diagonal and it is zero in the rest of the matrix.


% compute filter, using reverse correlation
% wsta = zeros(nx^2, 1);          % remove this line
xi = X(:,:,1);
xi = reshape(xi, nx^2, 1);
Sigma = mean(xi.^2); % mean(xi^2)
wsta = mean(rtilde .* Xtilde, 2) / Sigma^2;  % ex 1(c) compute wsta = <rx>/sigma^2 (a vector with dimensions nx^2 * 1)


% plot filter (reshaped to match patch)
figure('Name', 'filter')
imagesc(reshape(wsta, nx, nx))
axis square

% plot filter input versus firing rate
figure('Name', 'filter versus spikes')
hold on
plot(wsta'*Xtilde, r, 'k.');    % ex 1(d) scatter plot of filtered wsta*Xtilde versus spike count  
plot([-30, 40], [-30, 40], 'k--');
hold off
xlabel('wsta * x')
ylabel('spike count')

%% exercise 2: fit a linear-nonlinear-poisson (LNP) model

% initiliase parameters
b = 0;                      % initial bias term
w = 1e-9*randn(nx^2, 1);    % initial linear weights

% parameters for gradient descent
Nit = 300;                  % number of iterations
eta = 1e-2;                 % learning rate

L = zeros(1, Nit);          % log likelihood
for i = 1:Nit
    
    % firing rate  
    % f = ones(size(r));      % remove this line
    f = exp(w'*Xtilde+b);                % ex. 2(a) predicted firing rate on each trial (vector of size 1 * N)

    % log firing rate
    L(i) = (log(f)*r' - sum(f))/N;

    % derivative of log likelihood
    dL_w = Xtilde*(r - f)'/N;
    dL_b = sum(r-f)/N;

    % update parameters 
    w = w + eta * dL_w;                 % ex 2(b) gradient ascent updates of w and b
    b = b + eta * dL_b;
end

% plot loss function versus trials
figure('Name', 'Loss function')
plot(1:Nit, L)
xlabel('iterations')
ylabel('log likelihood')


figure('Name', 'learned  filter')
imagesc(reshape(w, nx, nx))
% imagesc(w)
axis square
% imagesc()                  % ex 2(c) plot filter


figure('Name', 'spike count versus prediction')
plot(exp(w'*Xtilde+b), r, 'k.'); hold on             % ex 2(d) compare predicted and observed spike count
% plot(f, rtilde, 'k.'); hold on             % ex 2(d) compare predicted and observed spike count
plot([0, 120], [0, 120], 'k--'); hold off
xlabel('predicted mean spike count')
ylabel('observed spike count')

% The LNP (Linear-Nonlinear-Poisson) model and reverse correlation are two methods commonly used in the analysis of neural responses to sensory stimuli. Here are some advantages of the LNP model over reverse correlation:

%     The LNP model is a more principled approach to modeling neural responses as it is based on a mathematical framework that describes the relationship between the stimulus and the response. In contrast, reverse correlation is a more empirical approach that does not explicitly model the neural mechanism.

%     The LNP model can be used to make predictions about neural responses to novel stimuli, whereas reverse correlation is limited to the stimuli that were presented during the experiment.

%     The LNP model can be used to estimate the contribution of different features of the stimulus to the neural response, whereas reverse correlation only provides an estimate of the receptive field of the neuron.

%     The LNP model can be extended to include nonlinearities beyond simple thresholding, whereas reverse correlation assumes that the neuron's response is a thresholded linear function of the stimulus.

% However, it is worth noting that both methods have their own strengths and limitations, and the choice of method depends on the specific research question and the nature of the data.


% The LNP (Linear-Nonlinear-Poisson) model is a powerful tool for modeling the relationship between stimuli and neural responses. However, there are several limitations to the model, including:

%     The LNP model assumes that the relationship between the stimulus and neural response is linear, which may not always be true. In many cases, the relationship may be more complex and non-linear, requiring more sophisticated models.

%     The model assumes that the response of the neuron is a Poisson process, which may not always be the case. For example, in the presence of strong correlations between the stimulus and response, the Poisson assumption may break down.

%     The LNP model requires a large amount of data to estimate the parameters of the model accurately. In practice, it may be challenging to obtain enough data to fit the model reliably.

%     The model does not take into account the dynamics of the neural response, such as adaptation and refractory effects, which can have a significant impact on the response.

%     The LNP model does not capture the full richness of the neural code. For example, the model does not account for the fact that neurons may encode information in the timing of spikes, as well as in the rate of firing.

% Overall, while the LNP model is a useful and widely used tool for modeling neural responses, it has several limitations that must be taken into account when interpreting its results.



%% exercise 3: compute information
f = exp(w'*Xtilde+b);                   % use answer to exercise 2(a)
% rpred = poissrnd(mean(r), 1, 2000); % ex 3(a) sample spike counts from model 
rpred = poissrnd(r); % ex 3(a) sample spike counts from model 




% histogram of spike counts
n_r = histcounts(rpred, 'BinMethod', 'integers'); 
% pr = ones(size(n_r));                   % remove this line
pr = (n_r - mean(n_r))/std(n_r);                               % ex 3(a) normalise histogram to obtain probability distribution p(r)
logpr = log(pr+eps);

% maximum number of spikes
rmax = 300;

% log p(r|x) - compute for each spike count (between 0 and rmax), and stimulus
logpr_x = zeros(rmax+1, N);
for k = 0:rmax    
    logpr_x(k+1, :) = k*log(f) - f - gammaln(k+1);
end
pr_x = exp(logpr_x);

%                                    ex 3(b): compute response entropy,
%                                    H(R), noise entropy H(R|X), and
%                                    information 
% HR = 0; HR_X = 0; I=0;               % remove this line

HR   = -sum(pr .* logpr)
HR_X = -mean(sum(pr_x .* logpr_x))
I = HR - HR_X        

% ! not the same result of multual information after each run because the
% histogram will separates the data each time differently


%% exercise 3 continued: how does mutual information vary with non-linearity?

%%% compute the non-linearity with 3 different values of b0
b0 = linspace(-3, 3, 30);

f = zeros(numel(b0), N);
A = 30;                         % amplitude of firing rate

% loop over different values of b0
for iter = 1:numel(b0)
    
    % f(iter, :) = A / (1 + exp(-w))             % ex 3(c) compute firing rate, with different values of b0

    % rpred = ?                 exp 3(d) sample from model

    n_r = histcounts(rpred, 'BinMethod', 'integers');
%     pr = ?                     % ex 3(d) normalise to obtain probability distribution p(r)        

    rmax = 100;
    logpr_x = zeros(rmax+1, N);
    for k = 0:rmax
        logpr_x(k+1, :) = k*log(f(iter, :)) - f(iter, :) - gammaln(k+1);
    end
    pr_x = exp(logpr_x);

    %%%%%%                      % exp 3(d) compute response entropy, H(R)
    %%%%%%                      and entropy H(R), and then compute mutual
    %%%%%%                      information
    % HR =                               
    % HR_X = 

%     I(iter) = ?
end


figure('Name', 'firing rate')
% plot(w'*Xtilde, f([1, 15, 30], :)', '.')   % ex 3(c): plot firing rate with different non-linearities 
xlabel('w*x')
ylabel('output')


figure('Name', 'Information')
% plot(                                % exp 3(e) plot information versus b0
xlabel('b0')
ylabel('information')




