function test_mpc(K)
  disp("simu1")
  simulateMPC([0.683;-0.864],K,0.5)
  hold on
  disp("simu2")
  simulateMPC([-0.523;0.244],K,0.5)
  hold on
  disp("simu3")
  simulateMPC([0.808;-0.121],K,0.5)
  hold on
  disp("simu4")
  simulateMPC([0.774;-0.222],K,0.5)
  hold on
  disp("simu5")
  simulateMPC([0.292;-0.228],K,0.5)
  hold on
  disp("simu6")
  simulateMPC([-0.08;-0.804],K,0.5)
end
