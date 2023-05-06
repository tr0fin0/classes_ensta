%%  2023_04_11 TP 3
%   MI210 - Neurocomputational Models
%   ===========================================================================

clc
clear all
close all


%%  creat data
%   ===========================================================================

%   stimulus coherences
coherence = [0.025 0.05 0.1 0.2 0.3 0.5 0.7];
%   coherence refers to the strength or degree of similarity between the stimuli
%   presented to the neuron:
%       - coherence of 0 indicates that the stimuli are completely dissimilar;
%       - coherence of 1 indicates that the stimuli are identical;

n_stimulus = numel(coherence);          % number of stimuli

mean_background = 10;                   % background mean spike count
lambda = mean_background+30*coherence;  % mean spike count 
n_trials = 1e3;                         % number of trials

spikes_0 = poissrnd(repmat(mean_background, n_trials, 1));  % generate spikes (0% coherence)
spikes  = poissrnd(repmat(lambda, n_trials, 1));            % generate spikes

%   as spikes_0 has 0% coherence data no spike should be produced by this stimulus



%%  plot histograms
%   ===========================================================================

x = 0:50;                  % different spike counts for histogram
edges = [x-0.5,x(end)+1]; % bin edges for histogram

hist_0 = histcounts(spikes_0, edges);            % histogram of spike counts (0% coherence)

figure('Name', 'Firing Rate Histogram')
for i = 1:n_stimulus
    hist_i = histcounts(spikes(:, i), edges);   % histogram of spike counts
   
    subplot(n_stimulus, 1, i) 
    hold on
    bar(x, hist_0)
    bar(x, hist_i)
    grid on
    ylabel('trials')
    title(sprintf('coherence = %.1f %%', coherence(i)*100));
end
xlabel('spike count')

%   we can see that the total of stimulus are the same but more spread between 
%   multiple spike counts.

%   more coherence means data closer to a true stimulus so spike count would increase.

%   as consequence the standard deviation of the stimulus would also increase because...



%%  Receiver Operator Characteristic, ROC, curves
%   ===========================================================================

%   according to chatGPT:
%   The receiver operating characteristic (ROC) curve is a graphical 
%   representation of the performance of a binary classifier as the 
%   discrimination threshold is varied. The ROC curve plots the true positive 
%   rate (TPR) against the false positive rate (FPR) for different threshold 
%   values. 

%   The area under the ROC curve (AUC) is a commonly used metric for 
%   quantifying the overall performance of the classifier, with an AUC of 1.0 
%   indicating perfect classification and an AUC of 0.5 indicating random 
%   classification.

z = 50:-1:0;                                        % thresholds
n_thresholds = numel(z);                            % number of thresholds

false_positive = zeros(n_thresholds, 1);            % false positive rate
true_positive  = zeros(n_thresholds, n_stimulus);   % hit rate

% loop over thresholds
for i = 1:n_thresholds
    false_positive(i)   = mean(spikes_0 > z(i));    % false positive rate
    true_positive(i, :) = mean(spikes  >= z(i));    % hit rate
    % this is a vetorial operation
end

%   plot ROC curve
figure('Name', 'Receiver Operator Characteristic')
hold on
plot(false_positive, true_positive, 'o-')
plot([0 1], [0 1], 'k--')
hold off
xlabel('false positive')
ylabel('true positive') 
legend('2.5%', '5.0%', '10.0%', '20.0%', '30.0%', '50.0%', '70.0%')
grid on


%%  Area Under Curve, AUC, and 2 Alternative Forced Choice, 2pAFC
%   performance in 2 alternative forced choice experiment 
%   ===========================================================================

%   The AUC provides an aggregate measure of performance across all possible
%   stimulus intensities. The AUC is computed by approximating the integral
%   under the ROC curve using a finite sum.

%   The second measure computed is the probability of correct response in the
%   2AFC experiment (p2AFC). This is simply the proportion of trials where the
%   subject responded correctly.

%   compute area under curve
dalpha = false_positive(2:end)-false_positive(1:end-1);
AUC = dalpha'*true_positive(1:end-1,:);

%   compute error in p2AFC
p2AFC = mean(spikes >= spikes_0);

%   we want to calculate both metrics in order to compare them and see if
%   them correctly represent a good approximation for the data.

%   where MATLAB does a vectorial operation comparing each colun with the
%   spikes_0 column that, in this case, would return boolean values if true or
%   not.

%   calculating the mean would return values between 0 and 1 that represent, 
%   in average, how good this estimator fits our data.


figure('Name', 'neuronal')
semilogx(100*coherence, AUC, '-o'); hold on
semilogx(100*coherence, p2AFC, 'r-o'); hold on
xlabel('log coherence')
ylabel('area under curve')
leg = legend('area under curve', 'probability correct');
set(leg, 'Location', 'SouthEast', 'Box', 'off', 'Fontsize', 12)

%   The strength of AUC is that it provides a measure of overall performance,
%   taking into account both sensitivity and specificity, and is not affected by
%   changes in the criterion. Moreover, AUC is a continuous measure, which makes
%   it possible to compare performance across different tasks and experimental
%   conditions.

%   On the other hand, AUC has some limitations. First, it does not provide
%   information about the optimal criterion level or the signal-to-noise ratio
%   of the system. Second, it assumes that the cost of false positives and false
%   negatives is equal, which may not be the case in all situations. Finally,
%   AUC requires a large number of trials to estimate accurately, which can be
%   time-consuming and resource-intensive.

%   The strength of 2AFC is that it provides a measure of sensitivity and
%   specificity separately, which can be useful for understanding the underlying
%   mechanisms of the system. 2AFC also has the advantage of being easy to
%   compute and interpret.

%   However, 2AFC also has some limitations. First, it requires the experimenter
%   to choose a criterion level, which can affect the outcome of the analysis.
%   Second, it can be affected by changes in the criterion level, which makes it
%   difficult to compare performance across different tasks and experimental
%   conditions. Finally, 2AFC assumes that the cost of false positives and false
%   negatives is equal, which may not be the case in all situations.


%%  compute entropy for binary stimulus
%   ===========================================================================

%   In information theory, entropy is a measure of the uncertainty or randomness
%   in a random variable. In the context of data, entropy is a measure of the
%   amount of information contained in the data or the degree of randomness in
%   the data.

%   Mathematically, the entropy of a discrete random variable X with possible
%   values {x1, x2, ..., xn} and probability mass function P(X) is defined as:

%       H(X) = -Σ P(xi) log2 P(xi)

%   where log2 is the base-2 logarithm, and the summation is taken over all
%   possible values of X.

%   The entropy is maximized when all possible values of X are equally likely
%   (i.e., a uniform distribution) and minimized when the distribution is
%   deterministic (i.e., all probability mass is concentrated on a single value
%   of X). The entropy is always non-negative and is measured in bits, nats, or
%   other units depending on the choice of the logarithm base.

p = linspace(0, 1, 100);                 % probability that x = 1
H =  -p.*log(p) - (1-p).*log(1-p);       % entropy

figure('Name', 'entropy of a binary stimulus')
plot(p, H)
xlabel('p') 
ylabel('H')
grid on






%% compute mutual information between stimulus and response
% log p(x)
% logpx = -repmat(??, 1, nstim);                TODO compute log p(x)

% maximum firing rate
rmax = 100;

% log p(r|x)
logpr_x = zeros(rmax+1, numel(coherence));
for k = 0:100
    % logpr_x(k+1, :) =                         TODO compute log p(r|x)
                                                % tip: use Gamma(k+1) =
                                                % gammaln(k+1)
end

% compute p(r, x)
% prx =                                         TODO compute p(r,x)

% compute p(r)
% pr =                                         TODO compute  p(r)

% compute HR
% HR =                                          TODO compute H(R)

% compute HR_X
% HR =                                          TODO compute H(R|X)

% compute Info
% Inf =                                          TODO compute I(R; X)


% fprintf('\n neuron encodes %.3f bits\n', Inf)


%% bonus question: vary firing rate, or background, or stim. distribution