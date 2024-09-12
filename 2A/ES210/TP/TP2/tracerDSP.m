[psdtur,w]=periodogram(phitur.Data,[],[],1/Te);
[psdresInt,w]=periodogram(phiresInt.Data,[],[],1/Te);
[psdresLQG,w]=periodogram(phiresLQG.Data,[],[],1/Te);

%% tracé des DSP échelle log en x
figure('Name','DSP avec échelle en x logarithmique','NumberTitle','off')
subplot(211)
loglog(w,psdtur,'r',w,psdresInt,'g',w,psdresLQG,'b')
xlabel('Fréquences (Hz)')
ylabel('(secondes d''arc)^2')
title('DSP : perturbation (rouge), phase résid. int. (vert), phase résid. LQG (bleu)')
grid
axis([2e-2 100, 1e-10 1])

subplot(212)
loglog(w,cumsum(psdresInt),'g',w, cumsum(psdtur),'r',w, cumsum(psdresLQG),'b')
xlabel('Fréquences (Hz)')
ylabel('(secondes d''arc)^2')
title('DSP cumulées : perturbation (rouge), phase résid. int. (vert), phase résid. LQG (bleu)')
grid
axis([2e-2 100, 1e-7 1])

%% tracé des DSP échelle linéaire en x
figure('Name','DSP avec échelle en x linéaire','NumberTitle','off')
subplot(211)
semilogy(w,psdtur,'r',w,psdresInt,'g',w,psdresLQG,'b')
xlabel('Fréquences (Hz)')
ylabel('(secondes d''arc)^2')
title('DSP : perturbation (rouge), phase résid. int. (vert), phase résid. LQG (bleu)')
grid
axis([2e-2 100, 1e-10 1])

subplot(212)
semilogy(w,cumsum(psdresInt),'g',w, cumsum(psdtur),'r',w, cumsum(psdresLQG),'b')
xlabel('Fréquences (Hz)')
ylabel('(secondes d''arc)^2')
title('DSP cumulées : perturbation (rouge), phase résid. int. (vert), phase résid. LQG (bleu)')
grid
axis([2e-2 100, 1e-7 1])
