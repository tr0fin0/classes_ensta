%%============================================================================
%%                    Main Code
%%============================================================================
clc
clear all
close all

ID = 217276;
feature('DefaultCharacterSet','UTF-8')

%   Parameters Estimation by Noisly Sinus
%   Import data from PC5 moodle files
filtreKalman = load('filtreKalman.mat');
algoEM       = load('algoEM.mat');


exercice1(filtreKalman)
exercice2(algoEM)