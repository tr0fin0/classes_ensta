%% creat fake data

% range of stimulus coherences
coherence = [0.025 0.05 0.1 0.2 0.3 0.5 0.7];
nstim = numel(coherence);                       % number of stimuli

r0 = 10;                                        % background mean spike count
lambda = r0+30*coherence;                       % mean spike count 
ntr = 1e3;                                      % number of trials
r0 = poissrnd(repmat(r0, ntr, 1));              % generate spikes (0% coherence)
r = poissrnd(repmat(lambda, ntr, 1));           % generate spikes


%% plot histograms

x = 0:50;                  % different spike counts for histogram
edges =  [x-0.5,x(end)+1]; % bin edges for histogram

%%%%                       % TODO generate histogram of spike counts (0% coherence)

figure('Name', 'firing rate histograms')
for i = 1:nstim
    %%%%                    % TODO generate histogram of spike counts
   
    subplot(nstim, 1, i) 
    bar(x, [n0; n])
    ylabel('trials')
    title(sprintf('coherence = %.1f %%', coherence(i)*100));
    
end
xlabel('spike count')

%% ROC curves
z = 50:-1:0;                            % thresholds
nz = numel(z);                          % number of thresholds

alpha = zeros(nz, 1);                   % false alarm rate
beta = zeros(nz, nstim);                % hit rate

% loop over thresholds
for i = 1:nz
    %%%%%%%% alpha(i) =                TODO false alarm rate
    %%%%%%%% beta(i)  =                TODO: compute hit rate
end

% plot ROC curve
figure('Name', 'ROC curve')
%%% plot( ... , 'o-')                   % TODO plot ROC curve
plot([0 1], [0 1], 'k--'); hold off
xlabel('alpha')
ylabel('beta')
 


%% area under curve and performance in 2 alternative forced choice experiment 

%%%%%%%%%%                           TODO: compute area under curve
% AUC = 

%%%                                 TODO: compute error in 2AFC
% p2AFC = 

figure('Name', 'neuronal')
% semilogx(100*coherence, AUC, '-o'); hold on
% semilogx(100*coherence, p2AFC, 'r-o'); hold on
xlabel('log coherence')
ylabel('area under curve')
leg = legend('area under curve', 'probability correct');
set(leg, 'Location', 'SouthEast', 'Box', 'off', 'Fontsize', 12)


%% compute entropy for binary stimulus

% compute probability (that x=1) and entropy

figure('Name', 'entropy of a binary stimulus')
% plot(p, H)
xlabel('p') 
ylabel('H')




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