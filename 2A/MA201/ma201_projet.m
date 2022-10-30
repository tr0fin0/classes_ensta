%%============================================================================
%%                    Main Code
%%============================================================================
clc
clear all
close all

ID = 217276;
feature('DefaultCharacterSet','UTF-8')

%   Kalman Filter Simple
%   Import data from project files
mesuresK1 = load('mesurestrajKalm1.mat');
mesuresK2 = load('mesurestrajKalm2.mat');
mesuresK3 = load('mesurestrajKalm3.mat');


exercice1(filtrePart);