%%  2023_04_11 TP 1
%   MI210 - Neurocomputational Models
%   ===========================================================================

clc
clear all   % ctrl+r:   comment
close all   % ctrl+t: uncomment

RA = 217276;

whos -file dataENSTA_Lect_1.mat % sees  data
%   Name             Size              Bytes  Class     Attributes

%   binnedOFF      100x599            479200  double              
%   binnedON       100x599            479200  double              
%   dt               1x1                   8  double    sampling time          
%   stim           599x1                4792  double    stimulus

load dataENSTA_Lect_1.mat   % loads data

[R, T] = size(binnedOFF);
%   where:
%       - R: repetitions;
%       - T: measures in time;

integration_time = 0.5;
number_bins = integration_time/dt;  % number of time bins that correspond to desired
                                    % 0.5 seconds of integration time


%%  COMPUTE AND VISUALIZE STIMULUS PSTH
%   Peri-Stimulus Time Histogram
%   ============================================================================

%   PSTH stands for "Peri-Stimulus Time Histogram", which is a graphical 
%   representation of the firing rate of a neuron or a population of neurons 
%   over time in response to a specific stimulus or event.

%   The PSTH  in response to a stimulus is created by:
%       - dividing a period of time into small bins;
%       - counting the number of spikes that occur within each bin;

%   The resulting counts are then normalized by the bin width and the number
%   of trials to obtain an estimate of the average firing rate over time.

%   Normalized count is called the Spike Density Function also known from it's
%   accronym.

%   separated data from machine learning
training_time = number_bins : floor(0.66*T);    % training time bins, 2/3
testing_time  = ceil(0.66*T) : T;               % testing  time bins, 1/3
size_training = numel(training_time);
size_testing  = numel(testing_time);

%   select dataset:
if true
    %   studying OFF cells
    training_data = mean(binnedOFF(:,training_time), 1);
    testing_data  = mean(binnedOFF(:,testing_time),  1);
    
else
    %   studying ON  cells
    training_data = mean(binnedON(:,training_time), 1);
    testing_data  = mean(binnedON(:,testing_time),  1);

end

% normalization of bins
psth_training = training_data/dt;
psth_testing  = testing_data/dt;


fig=figure;

subplot(2,1,1)
hold on
title('stimulus')
plot((1:T)*dt, stim, 'k', 'LineWidth', 2.0)
xlabel('Time [s]'); xlim([0,15]);
ylabel('Luminance [?]')
grid on

subplot(2,1,2)
hold on
title('PSTH')
bar(training_time*dt, psth_training)
plot(training_time*dt, psth_training,'LineWidth', 2.0)
bar(testing_time*dt,  psth_testing)
plot(testing_time*dt,  psth_testing, 'LineWidth', 2.0)
xlabel('Time [s]'); xlim([0,15]);
ylabel('Spiking Rate [Hz]')
grid on
legend('', 'training', '', 'testing')

%   what we can notice:
%       - y-axis has is typical unit: Spikes per Second, [Hz] 
%           also know as Normalized Firing Rate;
%       - integration_time is respected: PSTH only start after the
%           0.5s were gone
% 
%   we can notice that the PSTH is sligthly right shifted dude to the integration
%   time chosen with spiking rate higher when the luminance is low, with a threshold
%   of around 55. 


%%  STIMULUS PRE-PROCESSING
%   ============================================================================

%   The pre-processing of a stimulus can involve several steps depending on the 
%   nature of the stimulus and the goals of the experiment. However, some common 
%   steps involved in stimulus pre-processing include:
%       - filtering: low-pass filter, band-pass filter or high-pass filter;
%       - scaling: normalize within a certain range;
%       - sampling: ;
%       - pre-processing: ;

%   we will use scaling: aiming a mean of zero and standard deviation of one.
stim_scaled = (stim' - mean(stim))/std(stim);
time_range = number_bins : T;

%   visualization
fig=figure;

subplot(2,1,1)
hold on
title('stimulus')
plot((1:T)*dt, stim, 'k', 'LineWidth', 2.0)
xlabel('Time [s]'); xlim([0,15]);
ylabel('Luminance [?]')
grid on

subplot(2,1,2)
hold on
title('stimulus scaled')
plot((1:T)*dt, stim_scaled, 'k', 'LineWidth', 2.0)
xlabel('Time [s]'); xlim([0,15]);
ylabel('Luminance [?]')
grid on

%   as we can see the curve is the same but shifted in th y-axis so it's mean
%   is on zero and is standar déviation in one as it's values are now between, 
%   for the most part, -1 and +1.


stim_full = zeros([T number_bins]);

%   each column will have an itegration window

for tt = time_range
    stim_full(tt, :) = stim_scaled((tt-number_bins+1) : tt);
end


%%  STIMULUS AUTOCORRELATION
%   ============================================================================
%   Stimulus autocorrelation refers to the degree to which a stimulus is correlated
%   with itself over time. 

%   Autocorrelation can be described as the degree to which a signal is correlated 
%   with a delayed version of itself.

%   In the context of stimulus processing, the autocorrelation of a stimulus is often
%   used to quantify its temporal structure.


%   luminance of time is related with the luminance from the previous and next time
%   simplify calculation from the correlation calcules
%   using the linear mode to calcule the matrix

%   compute autocorrelation of stim_full over training times
stim_autocorrelation = stim_full' * stim_full;

fig=figure;
hold on
%   notice that the time delay can be modify:
time_delay = 1;
%   this is possible because the autocorrelation varies with time delay
plot( 0:number_bins-1, stim_autocorrelation(time_delay,:)/size_training )
title('stimulus autocorrelation')
xlabel('Time [s]')
ylabel('autocorrelation');
% ylim([0 1]);
grid on

%   we can see that the autocorrelation is decreasing as time passed with means that
%   the stimulus is less correlated as time passes but it seems that it will stabilise
%   at around 0.4.

% expected that the data is decreesing, overtime the correlation is getting slower
% STA, the average of the data slide 13 of TD2
% STA


%%  STA, Spike-Triggered Average
%   ============================================================================
%   The spike-triggered average (STA) is a commonly used method in neuroscience
%   to characterize the relationship between the spiking activity of a neuron 
%   and its input.

%   It is a method for calculating the average stimulus that occurs before a
%   neuron fires an action potential (spike).

%   remove the mean to analyse the behavior in time
mean_training_data = mean(training_data);
training_data_zero_mean = training_data - mean_training_data;

STA = training_data_zero_mean * stim_full(training_time, :);

filter_linear_simple = STA * inv( diag( diag( stim_autocorrelation ) ) );
%   where only the main diagonal autocorrelation values were consider because
%   they are most relevant ones.

%   by the way it reduces the computation effort and therefore increases performance

%   visualization
fig=figure;
hold on
title('linear filter')
plot((1-number_bins:0)*dt, filter_linear_simple) % comment
plot( [-dt*number_bins 0], [0 0], '--k')
xlabel('time [s]')
ylabel('wLin')
grid on

%   interpretation chatGPT:
%       wLin positive: an increase in the stimulus increase the firing probability
%       wLin negative: an increase in the stimulus reduce   the firing probability


%   normalizy by the diagonal part of the 
%   ignore the fact the facct thtat the situmulous is corrected for the moment and continue with the calculation

% what are seeing:
    % it is biphasical, has a positive and a negative part
    % it is off, not logic gate behavior
    % it does not start at zero, has delay to behavior
    % as the values has autocorrelation 
    % as long as we ignore the autocorrelation the graph will have this behavior
    % biologicaly the plot it does not sense
    


%%  Linear Prediction
%   ============================================================================
%
% f(t) = \sum_tau w(tau) * ( x(t-tau) -mean(x) ) + mean_training_data
% missing the normal deviation of the sitimula
%

%   where we will try to predict the spikes of neurons with a linear model
%   given by the following:
prediction_linear_simple =  filter_linear_simple * stim_full(testing_time, :)' + mean_training_data;
% slide 1 page 13
%   where * is, in this case, a convolution of the linear filter with the stimulus

%   is worth meantioning that the testing time shall be used.

%   normalizing bins
psth_prediction_linear_simple = prediction_linear_simple/dt;

%   visualization
fig=figure;
hold on
title('PSTH: linear model')
plot(testing_time*dt, psth_testing,'LineWidth', 1.0)
plot(testing_time*dt, psth_prediction_linear_simple,'LineWidth', 2.0)
xlabel('Time [s]');         xlim([10 15])
ylabel('Spiking Rate (Hz)');
legend('data', 'linear')
grid on

%   as we can see on the graph there are negative spiking rate which has no
%   physical meaning and therefore indicate problems with this estimator.

%   in order to improve this prediction we could trunquate it's values to
%   remove the negative part.

%   as an estimator we can see that it's spikes are in phase with the data
%   but it's magnitude is far from coherent, peaks and valleys are not well
%   fitted.

%   compute the performance of the linear prediction with the 
performance_prediction_linear_simple = corr(psth_testing', prediction_linear_simple')

%   visualization
fig=figure;
hold on
title('scatter plot')
scatter(psth_testing, psth_prediction_linear_simple)
plot([-150 150],[-150 150],'--k')
xlabel('PSTH')
ylabel('prediction linear')
axis square;
grid on;



%%  ReLU Truncation
%   ============================================================================
%   
%   ReLU (Rectified Linear Unit) truncation is a non-linear activation function
%   that is commonly used in neural networks. It works by setting all negative
%   input values to zero and passing through positive input values.

%   The term "truncation" refers to the fact that the function is essentially
%   "cut off" at zero, meaning that it does not allow negative values to pass through.

%   implement non-linearity, very common in the brain, in the eyes and in machine learning.
%   it is need to add some non linearity to aproximate the system

prediction_ReLU_simple = max(prediction_linear_simple, 0);

%   normalizing bins
psth_prediction_ReLU_simple = prediction_ReLU_simple/dt;

fig=figure;
hold on
title('PSTH: ReLU')
plot(testing_time*dt, psth_testing)
plot(testing_time*dt, psth_prediction_linear_simple)
plot(testing_time*dt, psth_prediction_ReLU_simple, 'LineWidth', 2.0)
xlabel('Time [s]');         xlim([10 15])
ylabel('Spiking Rate [Hz]');
legend('data', 'linear', 'ReLU')
grid on

%   compute the performance of the ReLU with respect of the original data
performance_prediction_ReLU = corr(psth_testing', prediction_ReLU_simple')

%   visualization
fig=figure;
hold on
title('scatter plot')
scatter(psth_testing, psth_prediction_ReLU_simple)
plot([-150 150],[-150 150],'--k')
xlabel('PSTH')
ylabel('ReLU')
axis square;
grid on;

%   as we can by the increase of the correlation between the ReLU prediction
%   the ReLU truncation improves the estimation and correct the biological
%   incosistence.



%%  Full Autocovariance 
%   ============================================================================
% 
% STA(tau) = \sum_t rTilde(t) * xTilde(t-tau)
% w(tau) =  STA * Inv(autoCov)
% mean_training_data = mean( r(t) )
%

filter_linear_full = STA * inv( stim_autocorrelation );
%   when we consider all the autocorrelation values we can see that the
%   variance on the graph increases because the system is more sensible
%   to other values.

%   visualization
fig=figure;
hold on
title('linear filter')
plot((1-number_bins:0)*dt, filter_linear_simple)
plot((1-number_bins:0)*dt, filter_linear_full)
plot( [-dt*number_bins 0],[0 0],'--k')
xlabel('Past time [s]')
ylabel('wLin')
grid on

prediction_linear_full =  filter_linear_full * stim_full(testing_time, :)' + mean_training_data;

%   normalizing bins
psth_prediction_linear_full = prediction_linear_full/dt;

%   compute the performance
performance_prediction_linear_full = corr(psth_testing', prediction_linear_full')


%   applying ReLU truncation
prediction_ReLU_full = max(prediction_linear_full, 0);

%   normalizing bins
psth_prediction_ReLU_full = prediction_ReLU_full/dt;

%   compute the performance
performance_prediction_ReLU_full = corr(psth_testing', prediction_ReLU_full')



%   visualization
fig=figure;
hold on
title('PSTH: linear full')
plot(testing_time*dt, psth_testing)
plot(testing_time*dt, psth_prediction_linear_simple)
plot(testing_time*dt, psth_prediction_ReLU_simple)
plot(testing_time*dt, psth_prediction_linear_full)
plot(testing_time*dt, psth_prediction_ReLU_full)
xlabel('Time [s]');         xlim([10 15]);
ylabel('Spiking Rate [Hz]');
legend('data', 'linear simple', 'ReLU simple', 'linear full', 'ReLU full')
grid on

%   as we can see with the correlation value considering all the autocorrelation
%   improved the prediction overall but it is still far from great.

%   it seems that the prediction is smooter than the others and therefore it does
%   not have a lot of negative values but it can neither show the peaks correctly.



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




