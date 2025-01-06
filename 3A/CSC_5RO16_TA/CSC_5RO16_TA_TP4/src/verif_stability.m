function ok=verif_stability(x_verif)
  try
    pkg load control
  end
  %parametres
  mu=0.5;
  u0=0;
  x10=0;
  x20=0;
  
  %matrice de poids
  Q=[0.5,0;0,0.5];
  R=[1];
  
  %TODO écrire les matrics A et B de la linéarisation Jacobienne
  %linéarisés jacobienne
  %A=
  
  %B=

  %TODO avec riccati, trouver une commande stabilisante
  %essai riccati : A'P+PA-PB inv(R) B'P + Q =0
  %[x, l, g] = 
  K=-g;
  
  %TODO calculer l'équation du système avec rebouclage
  %systeme rebouclage
  %Ak=

  %eigs(Ak);
  
  M=[-1,0;0,-1] - (Ak);
  %disp(det(M));
  
  %disp(eig(Ak));
  
  %TODO calculer la borne lambda et la borne alpha à 95 % de lambda
  %calcul de la borne, on retrouve bien celle de l'article
  %lambda=
  % borne a 95 % 
  %alpha=
  
  %TODO écrire les matrices de l'équation de Lyapunov et la résoudre pour obtenir la matrice P
  %matrice pour equation lyap
  %Al=
  %Bl=
  
  P=lyap(Al,Bl);   
  
  %verif lyap
  %Al*P + P*Al' + Bl;
  %eigs(P);
  
  %ici on calcul la borne du problème quadratique beta
  [x1,obj]=qp([0;0],-2*P,[],[],[],[-0.8;-0.8],[0.8;0.8],-2,K,2)
  beta=-obj
  
  %TODO écrire le test qui valide ou non si le point est dans la zone de stabilité du controleur  
  %ok = 
  
  
endfunction
