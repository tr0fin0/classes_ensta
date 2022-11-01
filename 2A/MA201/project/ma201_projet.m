%%============================================================================
%%                    Main Code
%%============================================================================
clc
clear all
close all

ID = 217276;
% feature('DefaultCharacterSet','UTF-58')   % not compatible with octave

%   Kalman Filter Simple
%   Import data from project files
mesuresK1 = load('mesurestrajKalm1.mat');
mesuresK2 = load('mesurestrajKalm2.mat');
mesuresK3 = load('mesurestrajKalm3.mat');


%   ========================================
%   we studied a simple tracking system of a radar and a target, details in the
%   project report.

exercice1(mesuresK1);
exercice1(mesuresK2);