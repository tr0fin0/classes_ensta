%% creat fake data

% range of stimulus coherences
coherence = [0.025 0.05 0.1 0.2 0.3 0.5 0.7];
nstim = numel(coherence);                       % number of stimuli

r0 = 10;                                        % background mean spike count
lambda = lambda0+30*coherence;                       % mean spike count 
ntr = 1e3;                                      % number of trials
r0 = poissrnd(repmat(r0, ntr, 1));          % generate spikes (0% coherence)
r = poissrnd(repmat(lambda, ntr, 1));           % generate spikes


%% plot histograms

x = 0:50;                  % different spike counts for histogram
edges =  [x-0.5,x(end)+1]; % bin edges for histogram

n0 = histcounts(r0, edges);            % histogram of spike counts (0% coherence)

figure('Name', 'firing rate histograms')
for i = 1:nstim
    n = histcounts(r(:, i), edges);   % histogram of spike counts
   
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
    alpha(i) = mean(r0>z(i));             % false alarm rate
    beta(i, :) = mean(r >= z(i));         % hit rate
end

% plot ROC curve
figure('Name', 'ROC curve')
plot(alpha, beta, 'o-', 'Linewidth', 2); hold on
plot([0 1], [0 1], 'k--'); hold off
xlabel('alpha')
ylabel('beta')
 

%% area under curve and performance in 2 alternative forced choice experiment 

% compute area under curve
dalpha = alpha(2:end)-alpha(1:end-1);
AUC = dalpha'*beta(1:end-1,:) ;

% compute error in 2AFC
p2AFC = mean(r >= r0);

figure('Name', 'neuronal')
semilogx(100*coherence, AUC, '-o'); hold on
semilogx(100*coherence, p2AFC, 'r-o'); hold on
xlabel('log coherence')
ylabel('area under curve')
leg = legend('area under curve', 'probability correct');
set(leg, 'Location', 'SouthEast', 'Box', 'off', 'Fontsize', 12)


%% compute entropy for binary stimulus

p = linspace(0, 1, 100);                 % probability that x = 1
H =  -p.*log(p) - (1-p).*log(1-p);       % entropy

figure('Name', 'entropy of a binary stimulus')
plot(p, H)
xlabel('p') 
ylabel('H')


%% compute mutual information between stimulus and response

% log p(x)
logpx = -repmat(log(nstim), 1, nstim);

% maximum firing rate
rmax = 100;

% log p(r|x)
logpr_x = zeros(rmax+1, numel(coherence));
for k = 0:rmax
    logpr_x(k+1, :) = k*log(lambda) - lambda - gammaln(1+k);
end

% compute p(r, x)
prx = exp(logpr_x+logpx);

% compute p(r)
pr = sum(prx, 2);

% compute HR
HR = - pr(:)'*log(pr(:));

% compute HR_X
HR_X = - prx(:)'*logpr_x(:);

% compute Info
Inf = HR - HR_X;

fprintf('\n neuron encodes %.3f nats\n', Inf)

%% vary stimulus distribution
gain = [5 30 80 200 300 450 600];

Inf = zeros(numel(gain), 1);
for i = 1:numel(gain)
    
    lambda_new = lambda0+gain(i)*coherence;                       % mean spike count 


    % maximum firing rate
    rmax = max(2*(gain(i)+10), 100);

    % log p(r|x)
    logpr_x = zeros(rmax+1, numel(coherence));
    for k = 0:rmax
        logpr_x(k+1, :) = k*log(lambda_new) - lambda_new - gammaln(1+k);
    end

    % compute p(r, x)
    prx = exp(logpr_x+logpx);

    % compute p(r)
    pr = sum(prx, 2);

    % compute HR
    HR = - pr(:)'*log(pr(:));
    HX = - exp(logpx(:))'*logpx(:);

    % compute HR_X
    HR_X = - prx(:)'*logpr_x(:);

    % compute Info
    Inf(i) = HR - HR_X;
   
end


figure('Name', 'Inf versus sigma(x)')
plot(gain+10, Inf, '-o'); hold on
plot([0, max(gain+10)], HX*[1 1]); hold off
xlabel('maximum spike count')
ylabel('Information (nats)')


%% vary background firing rate




