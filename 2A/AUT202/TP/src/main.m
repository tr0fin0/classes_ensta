%%                    Main Code
%%============================================================================

clc
clear all   % ctrl+r:   comment
close all   % ctrl+t: uncomment

RA = 217276;


%%  2023/03/08 - TP
%%============================================================================
[J, m, sig, g] = Q5();
w = ((m*g^2)/((1+sig)*J))^(1/4);

ref=0.0;
% [I1, I2] = P2(0.000001, 0, 0, 0);
% part2_simu
% print('-spart2_simu','-dpdf','model2.pdf')

% [I1, I2] = P20(0.1, 0, 0, 0);
% part20_simu
% print('-spart20_simu','-dpdf','model20.pdf')

% [I1, I2, A, B, C, K] = P3(0.01,0.1,0.01,0.1,inf);
% part3_simu
% print('-spart3_simu','-dpdf','model3.pdf')

% [I1, I2, A, B, C, Ko, Kw] = P30(0.65,0.65,0.02,0.02,w);
% part30_simu
% print('-spart30_simu','-dpdf','model30.pdf')


[I1, I2, A, B, C, K, L] = P4(0.6,0.6,0.2,0.2,0.4,0.4,0.5,0.5,w);
part4_simu
print('-spart4_simu','-dpdf','model4.pdf')

% [I1, I2, A, B, C, K, L] = P5(0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.02,true);
% part5_simu
% print('-spart5_simu','-dpdf','model5.pdf')


x10  = I1(1); x20  = I1(2); x30  = I1(3); x40  = I1(4);
x10h = I2(1); x20h = I2(2); x30h = I2(3); x40h = I2(4);

% eigenValues = Q10(A);
% COM = Q11(A, B);
% cond(COM, 1);
% eig(A-B*K);

OBS = Q19(A, C);
cond(OBS, 1)
% eig(A-L*C);