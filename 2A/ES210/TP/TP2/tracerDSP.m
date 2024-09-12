[psdtur,w]=periodogram(phitur.Data,[],[],1/Te);
[psdresInt,w]=periodogram(phiresInt.Data,[],[],1/Te);
[psdresLQG,w]=periodogram(phiresLQG.Data,[],[],1/Te);

%% trac� des DSP �chelle log en x
figure('Name','DSP avec �chelle en x logarithmique','NumberTitle','off')
subplot(211)
loglog(w,psdtur,'r',w,psdresInt,'g',w,psdresLQG,'b')
xlabel('Fr�quences (Hz)')
ylabel('(secondes d''arc)^2')
title('DSP : perturbation (rouge), phase r�sid. int. (vert), phase r�sid. LQG (bleu)')
grid
axis([2e-2 100, 1e-10 1])

subplot(212)
loglog(w,cumsum(psdresInt),'g',w, cumsum(psdtur),'r',w, cumsum(psdresLQG),'b')
xlabel('Fr�quences (Hz)')
ylabel('(secondes d''arc)^2')
title('DSP cumul�es : perturbation (rouge), phase r�sid. int. (vert), phase r�sid. LQG (bleu)')
grid
axis([2e-2 100, 1e-7 1])

%% trac� des DSP �chelle lin�aire en x
figure('Name','DSP avec �chelle en x lin�aire','NumberTitle','off')
subplot(211)
semilogy(w,psdtur,'r',w,psdresInt,'g',w,psdresLQG,'b')
xlabel('Fr�quences (Hz)')
ylabel('(secondes d''arc)^2')
title('DSP : perturbation (rouge), phase r�sid. int. (vert), phase r�sid. LQG (bleu)')
grid
axis([2e-2 100, 1e-10 1])

subplot(212)
semilogy(w,cumsum(psdresInt),'g',w, cumsum(psdtur),'r',w, cumsum(psdresLQG),'b')
xlabel('Fr�quences (Hz)')
ylabel('(secondes d''arc)^2')
title('DSP cumul�es : perturbation (rouge), phase r�sid. int. (vert), phase r�sid. LQG (bleu)')
grid
axis([2e-2 100, 1e-7 1])
