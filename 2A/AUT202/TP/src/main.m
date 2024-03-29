%%                    Main Code
%%============================================================================

clc
clear all   % ctrl+r:   comment
close all   % ctrl+t: uncomment

RA = 217276;


%%  2023/03/08 - TP
%%============================================================================
[J, m, sig, g] = Q5();
w       = ((m*g^2)/((1+sig)*J))^(1/4);
ref     = 0.2;
model   = 5;

if model == 2
    [I1, I2] = P2(0.001, 0, 0, 0);
    part2_simu
    print('-spart2_simu','-dpdf','model2.pdf')

elseif model == 20
    [I1, I2] = P20(0.1, 0, 0, 0);
    part20_simu
    print('-spart20_simu','-dpdf','model20.pdf')

elseif model == 3
    [I1, I2, A, B, C, K] = P3(0.01,0.01,0.1,0.1,inf);
    part3_simu
    print('-spart3_simu','-dpdf','model3.pdf')
    
    K
    eigK = eig(A-B*K)


elseif model == 30
    [I1, I2, A, B, C, Ko, Kw] = P30(0.02,0.02,0.02,0.02,w);
    part30_simu
    print('-spart30_simu','-dpdf','model30.pdf')
    
    Ko
    Kw
    eigKo = eig(A-B*Ko)
    eigKw = eig(A-B*Kw)


elseif model == 4
    [I1, I2, A, B, C, K, L] = P4(0.06,0.06,0.2,0.2,0.4,0.4,0.5,0.5,w);
    part4_simu
    print('-spart4_simu','-dpdf','model4.pdf')
    
    K
    L
    eigL = eig(A-L*C)


elseif model == 5
    [I1, I2, A, B, C, K, L] = P5(0.2,0.2,0.01,0.01,0.1,0.1,0.02,0.02,w);
    part5_simu
    print('-spart5_simu','-dpdf','model5.pdf')
else
    disp(['warning: model not defined ' num2str(model)])
end


x10  = I1(1); x20  = I1(2); x30  = I1(3); x40  = I1(4);
x10h = I2(1); x20h = I2(2); x30h = I2(3); x40h = I2(4);




if ( (model == 3) || (model == 30) )
    eigenValues = Q10(A);
    COM = Q11(A, B);
    cond(COM)
end

if ( (model == 4) || (model == 5) )
    OBS = Q19(A, C);
    cond(OBS)
    eig(A-L*C);
end

disp(['m' num2str(model), '_r' num2str(ref), '_s' num2str(x10), '_o' num2str(x10h)])