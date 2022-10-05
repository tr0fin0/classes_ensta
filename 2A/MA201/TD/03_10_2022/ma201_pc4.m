%%============================================================================
%%                    Main Code
%%============================================================================
clc
clear all
close all

ID = 217276;
feature('DefaultCharacterSet','UTF-8')

%   Parameters Estimation by Noisly Sinus
%   Import data from PC4 moodle files
filter      = load('filter.mat')
sinusoide   = load('sinusoide.mat')


theta = gaussMarkov(sinusoide)