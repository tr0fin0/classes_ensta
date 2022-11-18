%%============================================================================
%%                    Main Code
%%============================================================================
clc
clear all
close all

ID = 217276;
feature('DefaultCharacterSet','UTF-8')

%   Parameters Estimation by Noisly Sinus
%   Import data from PC6 moodle files
filtrePart = load('filtrePart.mat');


exercice1(filtrePart);