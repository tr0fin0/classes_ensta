%%                    Main Code
%%============================================================================

clc
clear all   % ctrl+r:   comment
close all   % ctrl+t: uncomment

RA = 217276;

%%============================================================================
whos -file dataENSTA_Lect_1.mat % sees  data
% Name             Size              Bytes  Class     Attributes

% binnedOFF      100x599            479200  double              
% binnedON       100x599            479200  double              
% dt               1x1                   8  double              
% stim           599x1                4792  double              

load dataENSTA_Lect_1.mat       % loads data

[R T] = size(binnedOFF);
integrationTime = 0.5/dt;


%% COMPUTE AND VISUALIZE PSTH AND STIMULUS
%%============================================================================
%
% r = mean( n(t,r) , r )
% psth = r/dt
%
timeTr = integrationTime:floor(0.66*T);
timeTe = ceil(0.66*T):T;
TTr = numel(timeTr);
TTe = numel(timeTe);


rTr = mean(binnedOFF(:,timeTr),1);
rTe = mean(binnedOFF(:,timeTe),1);

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
plot(timeTr*dt,psthTr,'LineWidth',2.0)
plot(timeTe*dt,psthTe,'LineWidth',2.0)
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')


%%  LINEAR MODEL
%%============================================================================

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
%%============================================================================
timeRange = integrationTime:T;

%%%%%%%%%%%%    represents codes to be changed
% different, transpose of stimilus
stimTilde = (stim - mean(stim))/std(stim);  % remove mean and normalize over the standard deviation
%%%%%%%%%%%%

fullStim = zeros([T integrationTime]);

for tt=timeRange
    fullStim(tt,:) = stimTilde((tt-integrationTime+1) :tt);
end


%% STIMULUS AUTOCORRELATION
%%============================================================================
% luminance of time is related with the luminance from the previous and next time
% simplify calculation from the correlation calcules
% using the linear mode to calcule the matrix
%%%%%%%%%%%%
% different
stimAutoCorr = fullStim' * fullStim;        % compute autocorrelation of fullStim over training times
%%%%%%%%%%%%

fig=figure;
hold on
plot( 0:integrationTime-1,stimAutoCorr(1,:)/TTr )
xlabel('Time (s)')
ylabel('Auto correlation')
set(gca,'Fontsize',16);
set(gca,'box','off')
% expected that the data is decreesing, overtime the correlation is getting slower
% STA, the average of the data slide 13 of TD2
% STA


%% STA = Spike-Triggered Average, and LINEAR FILTER
%%============================================================================
% formula to calculated
% STA(tau) = \sum_t rTilde(t) * xTilde(t-tau)
% w(tau) =  STA * Inv(autoCov) 
% b = mean( r(t) ) response
%
% But we will first ignore off diagonal elements in autoCorrelation
%   input is normalized normaly
%   output is not normalized normaly
%


%%%%%%%%%%%%
b = mean(rTr);  % mean of rTr
rTrTilde = rTr - b; % remove the mean to analyse the behavior of the curve in time
STA = rTrTilde * fullStim(timeTr, :);
%%%%%%%%%%%%

wLin = STA * inv( diag(diag( stimAutoCorr ) ) ); % normalizy by the diagonal part of the 
% ignore the fact the facct thtat the situmulous is corrected for the moment and continue with the calculation
% in theory as the code is 
% in exam a code would be given to be completed but the evaluation will take part as what is happennig with the code
% what are seeing:
    % it is biphasical, has a positive and a negative part
    % it is off, not logic gate behavior
    % it does not start at zero, has delay to behavior
    % as the values has autocorrelation 
    % as long as we ignore the autocorrelation the graph will have this behavior
    % biologicaly the plot it does not sense

fig=figure;
hold on
plot((1-integrationTime:0)*dt,wLin) % comment
plot( [-dt*integrationTime 0],[0 0],'--k')
xlabel('Past time (s)')
ylabel('wLin')
set(gca,'Fontsize',16);
set(gca,'box','off')


%% LINEAR PREDICTION
%%============================================================================
%
% f(t) = \sum_tau w(tau) * ( x(t-tau) -mean(x) ) + b
% missing the normal deviation of the sitimula
%


%%%%%%%%%%%%
fLin =  wLin * fullStim(timeTe, :)' + b; % compute prediction for testing times
% makes the matrix multiplication or the convolution cod
%%%%%%%%%%%%

% perfLinReg = corr(psthTe', fLin')

fig=figure;
hold on
plot(timeTe*dt,fLin/dt,'LineWidth',2.0)
plot(timeTe*dt,psthTe,'LineWidth',1.0)
xlim([10 15])
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')

%% WHAT IS HAPPENING ??
%
% Do a scatter plot of psth against prediction
%
% average number of spakes so it should be always positive
% trunquate when the values are negative to remove the problems discovered from the output result

fig=figure;
hold on
%%%%%%%%%%%%
% plot( ??? , ??? , '.','MarkerSize',12)
%%%%%%%%%%%%
plot([-100 100],[-100 100],'--k')
xlabel('PSTH')
ylabel('PREDICTION')
set(gca,'Fontsize',16);
set(gca,'box','off')


%% ReLU TRUNCATION
%
% ReLU(x) = max(x,0)
%

%%%%%%%%%%%%
% implement non-linearity, very common in the brain, in the eyes and in machine learning. it is need to add some non linearity to aproximate the system

fReLU = max(fLin, 0);
% different
% perfReLU = ???
%%%%%%%%%%%%

fig=figure;
hold on
%%%%%%%%%%%%
% different
% plot( ??? , ??? ,'LineWidth',2.0)
% plot( ??? ,psthTe,'LineWidth',1.0)
%%%%%%%%%%%%

xlim([10 15])
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')

%% INCLUDING FULL AUTOCOVARIANCE 
% 
% STA(tau) = \sum_t rTilde(t) * xTilde(t-tau)
% w(tau) =  STA * Inv(autoCov)
% b = mean( r(t) )
%
% And now with the full autoCorrelation
%

%%%%%%%%%%%%
% different
% wLinAC = STA * ????
%%%%%%%%%%%%

fig=figure;
hold on
plot((1-integrationTime:0)*dt,wLin,'LineWidth',2.0) % comment
plot((1-integrationTime:0)*dt,wLinAC,'LineWidth',2.0) % comment
plot( [-dt*integrationTime 0],[0 0],'--k')
xlabel('Past time (s)')
ylabel('wLin')
set(gca,'Fontsize',16);
set(gca,'box','off')

%%%%%%%%%%%%
% different
% fLinAC = ???
%%%%%%%%%%%%

% different >---------------------------------------------------------------<
% fReLUAC 
perfLinAC = corr(psthTe', fLinAC')

fig=figure;
hold on
plot(timeTe*dt,fLin/dt,'LineWidth',2.0)
plot(timeTe*dt,psthTe,'LineWidth',1.0)
plot(timeTe*dt,fLinAC/dt,'LineWidth',1.0)
xlim([10 15]);
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')


%% SMOOTHNESS REGULARIZATION
% teacher version will have the response: it will be shared. run the teacher code and the read the code of the teacher and after we willl discuss the plot to try to understand the code and comment as it will be demaned for the examen
% work with the student code and then work with the teacher version
% f(t) = \sum_tau w(tau) * ( x(t-tau) -mean(x) ) + b
% We seek to minimize 1/2 * \sum_t (r(t) - f(t) ).^2 + lambda/2 * w * Laplacian * w
%
% w * Laplacian * w = sum_tau (w(tau) - w(tau+1)).^2
%


lapl = 4*eye(integrationTime);
lapl = lapl - diag( ones([integrationTime-1 1]),1);
lapl = lapl - diag( ones([integrationTime-1 1]),-1);
lapl(1,1) = 2;
lapl(end,end) = 2;

fig=figure;
imagesc(lapl);
colorbar;

%% Let's try with an example regularization strength 

lambda = 10;

%%%%%%%%%%%%
% wLinReg = ??? 
%%%%%%%%%%%%

fig=figure;
hold on
plot((1-integrationTime:0)*dt,wLin,'LineWidth',2.0) % comment
plot((1-integrationTime:0)*dt,wLinAC,'LineWidth',2.0) % comment
plot((1-integrationTime:0)*dt,wLinReg,'LineWidth',2.0) % comment
plot( [-dt*integrationTime 0],[0 0],'--k')
xlabel('Past time (s)')
ylabel('wLin')
set(gca,'Fontsize',16);
set(gca,'box','off')

%%%%%%%%%%%%
% fLinReg = ???
%%%%%%%%%%%%

perfLinReg = corr(psthTe', fLinReg')

fig=figure;
hold on
plot(timeTe*dt,fLin/dt,'LineWidth',2.0)
plot(timeTe*dt,psthTe,'LineWidth',1.0)
plot(timeTe*dt,fLinAC/dt,'LineWidth',1.0)
plot(timeTe*dt,fLinReg/dt,'LineWidth',1.0)
xlim([10 15])
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')

%% LAMBDA OPTIMIZATION

lambdaRange = logspace(-2,4,30);
nLambda = numel(lambdaRange);
perfLambda = zeros([nLambda 1]);

for ll = 1:nLambda
    lambda=lambdaRange(ll);
    wLinReg = STA * inv( stimAutoCorr + lambda * lapl );
    fLinReg = wLinReg * fullStim(timeTe,:)' + b; % comment

    perfLambda(ll) =  corr(psthTe', fLinReg');
end


fig=figure;
hold on
plot(lambdaRange,perfLambda,'LineWidth',2.0)
set(gca,'XScale','log')
xlabel('Regularization strenght')
ylabel('Performance')
set(gca,'Fontsize',16);
set(gca,'box','off')

%% COMPUTE OPTIMAL W and ADD ReLU

[~,llBest] = max(perfLambda);

lambda = lambdaRange(llBest);
wLinReg = STA * inv( stimAutoCorr + lambda * lapl );
fLinReg = wLinReg * fullStim(timeTe,:)' + b; % comment

perfLinReg = corr(psthTe', fLinReg')

perfReLUReg = corr(psthTe', max(fLinReg,0)')

fig=figure;
hold on
plot((1-integrationTime:0)*dt,wLin,'LineWidth',2.0) % comment
plot((1-integrationTime:0)*dt,wLinAC,'LineWidth',2.0) % comment
plot((1-integrationTime:0)*dt,wLinReg,'LineWidth',2.0) % comment
plot( [-dt*integrationTime 0],[0 0],'--k')
xlabel('Past time (s)')
ylabel('wLin')
set(gca,'Fontsize',16);
set(gca,'box','off')



