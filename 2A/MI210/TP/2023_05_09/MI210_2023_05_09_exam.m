%%  2023_05_09 examen
%   MI210 - Neurocomputational Models
%   ===========================================================================

clear
clc
close all

load('data_exam.mat');


%%  introduction 

%   size of the stimulus
[nx, ~, N] = size(X);

% nx = size of image patch in one direction
% N =  number of trials

%%% plot one example stimulus
figure('Name', 'example stimulus')
imagesc(X(:, :, 1))
colormap('gray')

% plot one example stimulus
figure('Name', 'histogram of spike counts')
hist(r, 100)                          
xlabel('recorded spike counts')
ylabel('number')


%% exercise 1: compute spike triggered average

% reshape stimulus: number of pixels (nx^2) x number of trials (N)
Xvec = reshape(X, nx^2, N);

% remove mean from stimulus 
Xtilde = zeros(size(Xvec)); rtilde = zeros(size(r));      % remove this line
% Xtilde =                      % ex 1(a) remove mean from stimulus                
% rtilde =                      % ex 1(a) remove mean from firing rate

                                % ex 1(b) can you show that the stimulus is
                                % 'white' (see exercise)?

% compute filter, using reverse correlation
wsta = zeros(nx^2, 1);          % remove this line
%%% wsta =                      % ex 1(c) compute wsta = <rx>/sigma^2 (a vector with dimensions nx^2 * 1)


% plot filter (reshaped to match patch)
figure('Name', 'filter')
imagesc(reshape(wsta, nx, nx))

% plot filter input versus firing rate
figure('Name', 'filter versus spikes')
%%%%%%%%% plot(?, ?, 'k.');    % ex 1(d) scatter plot of filtered wsta*Xtilde versus spike count  
hold on
plot([-30, 40], [-30, 40], 'k--'); hold off
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
    f = ones(size(r));      % remove this line
    % f = ?;                % ex. 2(a) predicted firing rate on each trial (vector of size 1 * N)

    % log firing rate
    L(i) = (log(f)*r' - sum(f))/N;

    % derivative of log likelihood
    dL_w = Xtilde*(r - f)'/N;
    dL_b = sum(r-f)/N;

    % update parameters 
%     w =                   % ex 2(b) gradient ascent updates of w and b
%     b = 
end

% plot loss function versus trials
figure('Name', 'Loss function')
plot(1:Nit, L)
xlabel('iterations')
ylabel('log likelihood')


figure('Name', 'learned  filter')
% imagesc()                  % ex 2(c) plot filter


figure('Name', 'spike count versus prediction')
% plot(?, r, 'k.'); hold on             % ex 2(d) compare predicted and observed spike count
plot([0, 120], [0, 120], 'k--'); hold off
xlabel('predicted mean spike count')
ylabel('observed spike count')

%% exercise 3: compute information
% f = ?                    % use answer to exercise 2(a)
% rpred =                  % ex 3(a) sample spike counts from model 

% histogram of spike counts
n_r = histcounts(rpred, 'BinMethod', 'integers'); 
pr = ones(size(n_r));                   % remove this line
% pr =                                  % ex 3(a) normalise histogram to obtain probability distribution p(r)
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
HR = 0; HR_X = 0; I=0;               % remove this line

%%% HR   = ???
%%% HR_X = ???
% I = ??        


%% exercise 3 continued: how does mutual information vary with non-linearity?

%%% compute the non-linearity with 3 different values of b0
b0 = linspace(-3, 3, 30);

f = zeros(numel(b0), N);
A = 30;                         % amplitude of firing rate

% loop over different values of b0
for iter = 1:numel(b0)
    
    % f(iter, :) = ?             % ex 3(c) compute firing rate, with different values of b0

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




