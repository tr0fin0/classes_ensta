Te=0.005;
g=0.1;

load geminiTT
semilogx(ff,dsp_perturb,'r')
grid

vartur=var(phitur.Data)
varres_INT=var(phiresInt.Data)
varres_LQG=var(phiresLQG.Data)

% Instructions � rajouter � la figure obtenue par l'analyse lin�aire
hold on
semilogx(ff,dsp_perturb,'r')

tracerDSP
