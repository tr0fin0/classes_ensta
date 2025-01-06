function plot_zone()
  figure()
  for i=-2:0.1:2
    for j=-2:0.1:2
      if verif_stability([i;j])
        plot(i,j,'+');
        hold on
      endif
    endfor
  endfor
  
endfunction