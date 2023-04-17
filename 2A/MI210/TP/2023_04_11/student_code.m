%%  2023_04_11 TP 2
%   MI210 - Neurocomputational Models
%   ===========================================================================

clear
clc
close all


%%  data create (fake)
%   ===========================================================================

%   range of stimulus coherences
coherence = [0.025 0.05 0.1 0.2 0.3 0.5 0.7];
n_stimulus= numel(coherence);                   % number of stimulus

mean    = 10;                                   % mean spike count background 
lambda  = mean + 30*coherence;                  % mean spike count 
n_trials= 1e3;                                  % number of trials
spike_counts_0  = poissrnd(repmat(mean, n_trials, 1));      % generate spikes (0% coherence)
spike_counts    = poissrnd(repmat(lambda, n_trials, 1));    % generate spikes


%%  plot histograms
%   ===========================================================================

num_bins = 50;
bin_edges = 0:1:num_bins;

%   generate histogram of spike counts (0% coherence)
bin_counts_0 = histograma(spike_counts_0, n_trials, num_bins, 0, 50);
bin_counts_array = [];

%   generate histogram of spike counts
figure('Name', 'firing rate histograms')
for i = 1:n_stimulus
    data = spike_counts(:,i);
    bin_counts = histograma(data, n_trials, num_bins, 0, 50);

    subplot(n_stimulus, 1, i) 
    bar(bin_edges(1:end-1), [bin_counts_0, bin_counts], 'hist')
   
    ylabel('trials')
    title(sprintf('coherence = %.1f %%', coherence(i)*100));

    bin_counts_array = [bin_counts_array, bin_counts];
end
    
xlabel('spike count')


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

thresholds = num_bins-1:-1:0;
num_threshold = numel(thresholds);

% define counter arrays
AUC_array = [];
p2AFC_array = [];

%   plot ROC curve
figure('Name', 'ROC curve')
hold on

%   loop over thresholds
for j = 1:n_stimulus
    % define counter storage arrays
    FP_rate = zeros(num_threshold, 1);          % FP, False  Positive rate, alpha
    TP_rate = zeros(num_threshold, n_stimulus); % TP, True   Positive rate, beta

    % define counters
    FP_count = 0;
    TP_count = 0;

    % calculate FP and TP rates
    for i = 1:num_threshold
        if bin_counts_array(i, j) >= thresholds(i)
            TP_count = TP_count + 1;
        else
            FP_count = FP_count + 1;
        end
        
        FP_rate(i)   = FP_count / num_bins;
        TP_rate(i, j) = TP_count / num_bins;
    end

    % calculate AUC
    AUC_array = [AUC_array; trapz(FP_rate, TP_rate)];

    % calculate 2pAFC
    p2AFC_array = [p2AFC_array, TP_rate(find(FP_rate>=0.5, 1, 'first'), j)];


    plot(FP_rate, TP_rate(:,j), 'o-')
    legend_str{j} = sprintf('coherence = %.1f %%', coherence(j)*100);
end

plot([0 1], [0 1], 'k--'); hold off
grid on
axis square
legend(legend_str)
xlabel('\alpha, false positive')
ylabel('\beta, true positive')
 


%%  Area Under Curve, AUC, and 2 Alternative Forced Choice, 2pAFC
%   performance in 2 alternative forced choice experiment 
%   ===========================================================================

figure('Name', 'neuronal AUC')
hold on
semilogx(100*coherence, AUC_array, '-o');
semilogx(100*coherence, p2AFC_array, 'r-o');
hold off
xlabel('log coherence')
ylabel('area under curve')
grid on
legend_str{end+1} = sprintf('p2AFC');
legend(legend_str);


%%  compute entropy for binary stimulus
%   ===========================================================================

%   compute probability (that x=1) and entropy

%   defining array of probabilities
p = 0:0.001:1;
H = -p .* log2(p) - (1-p) .* log2(1-p);

figure('Name', 'entropy of a binary stimulus')
plot(p, H)
grid on
axis square
legend('H(X)')
xlabel('p') 
ylabel('H')
%   we can notice that the probability of p(x=1) gives H(X=x)=0 which is
%   expected.

%   if we know the value of x from the start there is no uncertainty so
%   the entropy, randomness of the data, is zero.



%%  compute mutual information between stimulus and response
%   ===========================================================================
% log p(x)
% logpx = -repmat(??, 1, n_stimulus);                TODO compute log p(x)

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