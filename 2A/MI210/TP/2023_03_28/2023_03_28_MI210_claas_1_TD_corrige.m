clear all;
close all;

whos -file dataENSTA_Lect_1.mat
load dataENSTA_Lect_1.mat


[R T] = size(binnedOFF); % extract number of repetitions and number of time bins per repetition
integrationTime = 0.5/dt; % set the past-stimulus integration time (0.5sec) as number of time bins

% Define matlab color codes
matCol = [ 0    0.4470    0.7410; 
    0.8500    0.3250    0.0980;
    0.9290    0.6940    0.1250;
    0.4940    0.1840    0.5560;
    0.4660    0.6740    0.1880;
    0.3010    0.7450    0.9330;
    0.6350    0.0780    0.1840];

%% COMPUTE AND VISUALIZE PSTH AND STIMULUS
%
% r = mean( n(t,r) , r )
% psth = r/dt
%

timeTr = integrationTime:floor(0.66*T); % time bins for training
timeTe = ceil(0.66*T):T; % time bins for testing
TTr = numel(timeTr); % length of training
TTe = numel(timeTe); % length of testing

% for studying OFF cell
rTr = mean(binnedOFF(:,timeTr),1);
rTe = mean(binnedOFF(:,timeTe),1);

% for studying ON cell
%rTr = mean(binnedON(:,timeTr),1);
%rTe = mean(binnedON(:,timeTe),1);

% compute psth as rate in Hz
psthTr = rTr/dt;
psthTe = rTe/dt;

fig=figure;

subplot(2,1,1)
hold on
plot((1:T)*dt,stim,'k','LineWidth',2.0)
xlabel('Time (s)')
ylabel('Luminance')
set(gca,'Fontsize',16);
set(gca,'box','off')

subplot(2,1,2)
hold on
plot(timeTr*dt,psthTr,'LineWidth',2.0,'Color',matCol(1,:))
plot(timeTe*dt,psthTe,'LineWidth',2.0,'Color',matCol(2,:))
text(0.01,0.95,'Training','Units','normalized','HorizontalAlignment','left','Color',matCol(1,:),'FontSize',18);
text(0.99,0.95,'Testing','Units','normalized','HorizontalAlignment','right','Color',matCol(2,:),'FontSize',18);
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')

%%  LINEAR MODEL

% f(t) = \sum_tau w(tau) * ( x(t-tau) -mean(x) ) + b
% We seek to minimize 1/2 * \sum_t (r(t) - f(t) ).^2
%
% r(t) -> rTilde(t) = r(t)-mean(r)
% x(t) -> xTilde(t) = ( x(t)-mean(x) ) / std(x)
%
% autoCov(tau,tau') = \sum_t xTilde(t+tau) xTilde(t+tau')
% STA(tau) = \sum_t rTilde(t) * xTilde(t-tau)
% w(tau) =  STA * Inv(autoCov) 
% 



%% STIMULUS PRE-PROCESSING


timeRange = integrationTime:T; % exclude the first time bins for training

stimTilde = ( stim' - mean(stim) ) / std(stim); % normalize stimulus
fullStim = zeros([T integrationTime]); % for convenience expand stimulus fullStim(t,tau) = stim(t-tau+1)

for tt=timeRange
    fullStim(tt,:) = stimTilde((tt-integrationTime+1) :tt);
end

%% STIMULUS AUTOCORRELATION

stimAutoCorr = fullStim(timeTr,:)' * fullStim(timeTr,:); % compute stimulus autocorrelation

fig=figure;
hold on
plot( 0:integrationTime-1,stimAutoCorr(1,:)/TTr,'Linewidth',2.0,'Color',matCol(1,:) )
xlabel('Time (s)')
ylabel('Auto correlation')
ylim([0 1]);
set(gca,'Fontsize',16);
set(gca,'box','off')
% Do you understand this plot ?


%% STA = Spike-Triggered Average, and LINEAR FILTER
%
% STA(tau) = \sum_t rTilde(t) * xTilde(t-tau)
% w(tau) =  STA * Inv(autoCov) 
% b = mean( r(t) )
%
% But we will first ignore off diagonal elements in autoCorrelation
%
%

b = mean(rTr); % set the constant of the model
rTrTilde = rTr - b; % remove the mean from the response

STA = rTrTilde * fullStim(timeTr,:); % compute STA
wLin = STA * inv( diag(diag( stimAutoCorr ) ) ); % for the moment we ignore the offdiagonal autocorrelation

fig=figure;
hold on
plot((1-integrationTime:0)*dt,wLin,'Linewidth',2.0 ,'Color',matCol(3,:))
plot( [-dt*integrationTime 0],[0 0],'--k')
xlabel('Past time (s)')
ylabel('wLin')
set(gca,'Fontsize',16);
set(gca,'box','off')
% Do you understand this plot ?

%% LINEAR PREDICTION

% f(t) = \sum_tau w(tau) * ( x(t-tau) -mean(x) ) + b

fLin = wLin * fullStim(timeTe,:)' + b; % compute linear prediction

['perf lin. model = ' num2str( corr(psthTe', fLin'))]

fig=figure;
hold on
plot(timeTe*dt,psthTe,'LineWidth',2.0,'Color',matCol(2,:))
plot(timeTe*dt,fLin/dt,'LineWidth',1.0,'Color',matCol(3,:))
xlim([10 15])
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')

%% WHAT IS HAPPENING ??

% Do a scatter plot of psth against prediction

fig=figure;
hold on
plot(psthTe,fLin/dt,'.','MarkerSize',12)
plot([-100 100],[-100 100],'--k')
xlabel('PSTH')
ylabel('PREDICTION')
set(gca,'Fontsize',16);
set(gca,'box','off')
% Do you understand this plot ?

%% ReLU TRUNCATION
%
% ReLU(x) = max(x,0)
%
% fReLU = ReLU( fLin )
%


fReLU = max(fLin,0);
['perf lin. Model with ReLU = ' num2str(  corr(psthTe', fReLU'))]

fig=figure;
hold on
plot(timeTe*dt,fReLU/dt,'LineWidth',1.0,'Color',matCol(3,:))
plot(timeTe*dt,psthTe,'LineWidth',2.0,'Color',matCol(2,:))
xlim([10 15])
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')

%% INCLUDING AUTOCOVARIANCE ?
%
% STA(tau) = \sum_t rTilde(t) * xTilde(t-tau)
% w(tau) =  STA * Inv(autoCov)
% b = mean( r(t) )
%
% And now with the full autoCorrelation
%

wLinAC = STA * inv( stimAutoCorr ); % Using the full autocovariance matrix

fig=figure;
hold on
plot((1-integrationTime:0)*dt,wLin,'LineWidth',2.0,'Color',matCol(3,:)) % comment
plot((1-integrationTime:0)*dt,wLinAC,'LineWidth',2.0,'Color',matCol(4,:)) % comment
plot( [-dt*integrationTime 0],[0 0],'--k')
xlabel('Past time (s)')
ylabel('wLin')
set(gca,'Fontsize',16);
set(gca,'box','off')
% Do you understand this plot ?

fReLUAC = max( wLinAC * fullStim(timeTe,:)' + b , 0 ); % compute prediction

['perf Lin. model complete with ReLU = ' num2str( corr(psthTe', fReLUAC') )]

fig=figure;
hold on
plot(timeTe*dt,psthTe,'LineWidth',2.0,'Color',matCol(2,:))
plot(timeTe*dt,fReLU/dt,'LineWidth',1.0,'Color',matCol(3,:))
plot(timeTe*dt,fReLUAC/dt,'LineWidth',1.0,'Color',matCol(4,:))
xlim([10 15]);
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')



%% LN MODEL FITTING
%
% f(x) = exp( \sum_tau w(tau) * x(t-tau) +b)
% log p(r|x) = log Poiss(r|x) = r * log(f(x)) - f(x) 
%

% initiliase parameters
wLn = 1e-9*randn(1, integrationTime);
bLn = 0;

% parameters for gradient descent
Nit = 500;                  % number of iterations
eta = 1e-1;                 % step size

Ltr = zeros(1, Nit);          % log likelihood training
Lte = zeros(1, Nit);          % log likelihood testing
for i = 1:Nit
    
    % firing rate prediction
    fLNtr = exp(  wLn * fullStim(timeTr,:)' + bLn );
    fLNte = exp(  wLn * fullStim(timeTe,:)' + bLn );
    
    % log-likelihood training
    ll = log(fLNtr) .* rTr - fLNtr;
    Ltr(i) = mean(ll);

    % log-likelihood testing
    ll = log(fLNte) .* rTe - fLNte;
    Lte(i) = mean(ll);

    % derivative of log likelihood
    dL_w = ( rTr - fLNtr ) * fullStim(timeTr,:) /TTr;
    dL_b = sum(rTr-fLNtr)/TTr;

    % update parameters 
     wLn = wLn + eta * dL_w; 
     bLn = bLn + eta * dL_b;

end

fig=figure;
hold on
plot(1:Nit,Ltr,'--','LineWidth',2.0,'Color',matCol(5,:))
plot(1:Nit,Lte,'LineWidth',2.0,'Color',matCol(5,:))
%xlim([10 15])
xlabel('Epoch')
ylabel('Log Likelihood')
set(gca,'Fontsize',16);
set(gca,'box','off')

%% INFERRED FILTER, PREDICTION AND PERFORMANCE

fig=figure;
hold on
plot((1-integrationTime:0)*dt,wLin,'LineWidth',2.0,'Color',matCol(3,:)) % comment
plot((1-integrationTime:0)*dt,wLinAC,'LineWidth',2.0,'Color',matCol(4,:)) % comment
plot((1-integrationTime:0)*dt,wLn,'LineWidth',2.0,'Color',matCol(5,:)) % comment
plot( [-dt*integrationTime 0],[0 0],'--k')
xlabel('Past time (s)')
ylabel('wLin')
set(gca,'Fontsize',16);
set(gca,'box','off')
% How does it look?

fLNte = exp(  wLn * fullStim(timeTe,:)' + bLn );

['perf LN model = ' num2str( corr(psthTe', fLNte' ))]

fig=figure;
hold on
plot(timeTe*dt,psthTe,'LineWidth',2.0,'Color',matCol(2,:))
plot(timeTe*dt,fReLUAC/dt,'LineWidth',1.0,'Color',matCol(4,:))
plot(timeTe*dt,fLNte/dt,'LineWidth',1.0,'Color',matCol(5,:))
xlim([10 15])
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')



%% SMOOTHNESS REGULARIZATION

% f(t) = \sum_tau w(tau) * ( x(t-tau) -mean(x) ) + b
% We seek to minimize 1/2 * \sum_t (r(t) - f(t) ).^2 + lambda/2 * w * Laplacian * w
%
% w * Laplacian * w = sum_tau (w(tau) - w(tau+1)).^2
%


% let's define the laplacian matrix
lapl = 4*eye(integrationTime);
lapl = lapl - diag( ones([integrationTime-1 1]),1);
lapl = lapl - diag( ones([integrationTime-1 1]),-1);
lapl(1,1) = 2;
lapl(end,end) = 2;

fig=figure;
imagesc(lapl);
colorbar;



%% LN MODEL FITTING WITH SMOOTHNESS REGULARIZATION

% set regularization strenght value
lambda = 0.015; % this should be optimized over a validation set


% initiliase parameters
wLnReg = 1e-9*randn(1, integrationTime);
bLnReg = 0;

% parameters for gradient descent
Nit = 500;                  % number of iterations
eta = 1e-1;                 % step size

LtrReg = zeros(1, Nit);          % log likelihood training
LteReg = zeros(1, Nit);          % log likelihood testing
for i = 1:Nit
    
    % firing rate prediction
    fLNtrReg = exp(  wLnReg * fullStim(timeTr,:)' + bLnReg );
    fLNteReg = exp(  wLnReg * fullStim(timeTe,:)' + bLnReg );
    
    % log-likelihood training
    ll = log(fLNtrReg) .* rTr - fLNtrReg ;
    LtrReg(i) = mean(ll)  - 0.5 * lambda * wLnReg * lapl * wLnReg';

    % log-likelihood testing
    ll = log(fLNteReg) .* rTe - fLNteReg;
    LteReg(i) = mean(ll);

    % derivative of log likelihood
    dL_w = ( rTr - fLNtrReg ) * fullStim(timeTr,:) /TTr;
    dL_b = sum(rTr-fLNtrReg)/TTr;

    % update parameters 
     wLnReg = wLnReg + eta * dL_w - lambda * wLnReg * lapl; 
     bLnReg = bLnReg + eta * dL_b;

end

fig=figure;
hold on
plot(1:Nit,Ltr,'--','LineWidth',2.0,'Color',matCol(5,:))
plot(1:Nit,Lte,'LineWidth',2.0,'Color',matCol(5,:))
plot(1:Nit,LtrReg,'--','LineWidth',2.0,'Color',matCol(6,:))
plot(1:Nit,LteReg,'LineWidth',2.0,'Color',matCol(6,:))
%xlim([10 15])
xlabel('Epoch')
ylabel('Log Likelihood')
set(gca,'Fontsize',16);
set(gca,'box','off')

%% INFERRED FILTER, PREDICTION AND PERFORMANCE

fig=figure;
hold on
plot((1-integrationTime:0)*dt,wLin,'LineWidth',2.0,'Color',matCol(3,:)) % comment
plot((1-integrationTime:0)*dt,wLinAC,'LineWidth',2.0,'Color',matCol(4,:)) % comment
plot((1-integrationTime:0)*dt,wLn,'LineWidth',2.0,'Color',matCol(5,:)) % comment
plot((1-integrationTime:0)*dt,wLnReg,'LineWidth',2.0,'Color',matCol(6,:)) % comment
plot( [-dt*integrationTime 0],[0 0],'--k')
xlabel('Past time (s)')
ylabel('wLin')
set(gca,'Fontsize',16);
set(gca,'box','off')
% How does it look?


['perf LN model with Reg = ' num2str( corr(psthTe', fLNteReg' ))]

fig=figure;
hold on
plot(timeTe*dt,psthTe,'LineWidth',2.0,'Color',matCol(2,:))
plot(timeTe*dt,fReLUAC/dt,'LineWidth',1.0,'Color',matCol(4,:))
plot(timeTe*dt,fLNte/dt,'LineWidth',1.0,'Color',matCol(5,:))
plot(timeTe*dt,fLNteReg/dt,'LineWidth',1.0,'Color',matCol(6,:))
xlim([10 15])
ylim([0 200])
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')

%
%
%
% Are you satisfied with it?
% What can you do next?
%
%
%
% Exercize: try with different non-linearities: 
% ReLU(x) ? 
% Softmax = log( 1+exp(x) ) ?
%
%
%

