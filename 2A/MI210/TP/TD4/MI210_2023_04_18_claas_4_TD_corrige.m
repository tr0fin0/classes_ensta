%%%%%%%%% Exercise 1: compute entropy of 2 neurons, with varying covariance matrices  %%%%%%%

for i = 1:3
    
if i==1
     Sigma = [1 0; 0 1];
elseif i==2
    Sigma = [1 0.5; 0.5 1];
elseif i==3
    Sigma = [1 0.99; 0.99 1];
end
    H = 0.5*(log(2*pi*exp(1)*det(Sigma )))
end


%%  Load data, x: 16000 image patches of size 12 x 12 = 144

load('data.mat')


%% %%%%%%%%%%% Exercise 2 Compute the covariance of x 
C  = x*x'/size(x, 2);

z = WZ*x;

Czz = z*z'/size(z, 2);

figure('Name', 'covariance')
subplot(1, 2, 1)
imagesc(C(1:12, 1:12)); 
colormap('gray')
title('stimulus covariance')
subplot(1, 2, 2)
imagesc(Czz(1:12, 1:12)); 
colormap('gray')
title('response covariance')

%% convolve images with filters
w = reshape(WZ(78, :), 12, 12);

Z = conv2(X, w, 'same');

Xn = X + 0.7*std(X(:))*randn(size(X));

Zn = conv2(Xn, w, 'same');

figure('Name', 'images')
subplot(2, 2, 1)
imagesc(X);
subplot(2, 2, 2)
imagesc(Z, [-1 1]);
subplot(2, 2, 3)
imagesc(Xn);
subplot(2, 2, 4)
imagesc(Zn, [-1 1]);
colormap('gray')


%% Exercise 4: denoising filter
eta = [0 0.3 1.4]*std(x(:));
figure('Name', 'filters at different noise level')
for i = 1:3
    Wn = (C + eta(i)^2*eye(144))\C;

    Wcombined = WZ*Wn;

    wcombined = reshape(Wcombined(:, 78), 12, 12);
    
    subplot(1, 3, i)
    imagesc(wcombined); axis square; axis off;
end


%% Exercise 5:
eta = 0.7*std(x(:));

Wn = (C + eta^2*eye(144))\C;

Wcombined = WZ*Wn;

w = reshape(Wcombined(78, :), 12, 12);

Z = conv2(X, w, 'same');

Xn = X + eta*randn(size(X));

Zn = conv2(Xn, w, 'same');

figure('Name', 'images')
subplot(2, 2, 1)
imagesc(X);
subplot(2, 2, 2)
imagesc(Z, [-1 1]);
subplot(2, 2, 3)
imagesc(Xn);
subplot(2, 2, 4)
imagesc(Zn, [-1 1]);
colormap('gray')



%%
figure('Name', 'ICA filters')
for i = 1:25
    subplot(5, 5, i)
    imagesc(reshape(WI(i, :), 12, 12)); colormap('gray'); axis off; axis square
end

%%
grd = -100:1:100;

Z = WZ*x;
ZICA = WI*x;

pz = hist(Z(:), grd);
pr = hist(ZICA(:), grd);

figure('Name', 'responses')
semilogy(grd, pz, 'k'); hold on
semilogy(grd, pr, 'r'); hold on
legend('decorrelated', 'independent component')

%% 

pz1z2 = hist3(ZICA([9, 10], :)', {grd' grd'});
pz2 = sum(pz1z2, 1);

pz1_z2 = pz1z2./(pz2+1e-5);

figure('Name', 'conditional histogram')
imagesc(grd, grd, log(pz1_z2+0.01))
set(gca, 'Xlim', [-10 10], 'Ylim', [-10 10])
colormap('gray')
