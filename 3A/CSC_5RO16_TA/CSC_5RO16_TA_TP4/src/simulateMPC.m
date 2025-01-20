function simulateMPC(xinit,K,mu)

    dt = 0.01;
    x = xinit;
    xstore = NaN * zeros(2,10000);
    k = 1;
    u = 1;
    n = 4;

    while (norm(x) > 0.001) && (k<2000)
        xstore(:,k) = x;


        %TODO linearisation en x et t
        A = [  u * (1 - mu), 1; 1, - u * 4 * (1 - mu) ];
        B = [ mu + (1 - mu) * x(1); mu - 4 * (1 - mu) * x(2) ];

        A = A*dt + eye(2, 2);
        B = B*dt;

        %vecteur d'entrées
        U=[u,u,u,u]';
        H=eye(n,n)*2;

        %TODO écrire les matrices de la commande prédictive linéaire
        A_hat = [...
            A;...
            A^2;...
            A^3;...
            A^4];
        B_hat = [...
            B, zeros(2, n-1);...
            A^1 * B, B, zeros(2, n-2);...
            A^2 * B, A^1 * B, B, zeros(2, n-3);...
            A^3 * B, A^2 * B, A^1 * B, B];

        %TODO avec une pseudo inverse, calculer le vecteur d'entrées
        U = pinv(B_hat) * (-A_hat * x);


        if size(K) == 0
          u = U(1);
        else
          u = -K*x;
        endif

        if (u > 2)
          u = 2;
        elseif (u<-2)
          u = -2;
        endif

        %simu avec euler
        x1 = x(1);
        x2 = x(2);
        x(1) = x1 + dt*(x2 + u*(mu + 1*(1-mu)*x1));
        x(2) = x2 + dt*(x1 + u*(mu - 4*(1-mu)*x2));

        k++;
    endwhile

    if norm(x) < 0.01
        plot(xstore(1,:),xstore(2,:),'+');
    else
        disp("fail!")
    endif
endfunction
