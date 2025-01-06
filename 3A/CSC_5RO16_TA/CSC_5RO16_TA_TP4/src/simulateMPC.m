function simulateMPC(xinit,K,mu)
  
  dt=.01;
  x=xinit;
  xstore = NaN*zeros(2,10000);
  k=1;
  u=1;
  
  n=4;
  while (norm(x) > 0.001) && (k<2000)
    xstore(:,k) = x;
    
    
    %TODO linearisation en x et t
    %A=
    %B=
    
    %vecteur d'entrées
    U=[u,u,u,u]';
    H=eye(n,n)*2;
    
    %TODO écrire les matrices de la commande prédictive linéaire
    %Aqp=
    
    %Bqp=
      
    %TODO avec une pseudo inverse, calculer le vecteur d'entrées
    %U=
    
    
    if size(K)==0
      u=U(1);
    else
      u=-K*x;
    end
    
    if (u > 2)
      u=2;
    elseif (u<-2)
      u=-2;
    end
    
    %simu avec euler
    x1=x(1);
    x2=x(2);
    x(1) = x1 + dt*(x2 + u*(mu + (1-mu)*x1));
    x(2) = x2 + dt*(x1+u*(mu-4*(1-mu)*x2));
    
    k++;
  end
  
  if norm(x) < 0.01
    plot(xstore(1,:),xstore(2,:),'+');
  else
    disp("fail!")
  end
endfunction
