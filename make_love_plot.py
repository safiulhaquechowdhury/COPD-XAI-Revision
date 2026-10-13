import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats; import pandas as pd
R=json.load(open('results/results.json')); S=json.load(open('results/supplementary_results.json'))
bal=R['psm']['balance']; names=[b['covariate'] for b in bal][::-1]
f,ax=plt.subplots(figsize=(5.4,4.2)); y=np.arange(len(names))
ax.scatter([abs(b['smd_before']) for b in bal][::-1],y,marker='o',color='#C44E52',label='Before matching',zorder=3)
ax.scatter([abs(b['smd_after']) for b in bal][::-1],y+0.15,marker='s',color='#4C72B0',label='After 1:1 matching',zorder=3)
ax.scatter([abs(S['ipw_smd'][n]) for n in names],y-0.15,marker='^',color='#55A868',label='After weighting',zorder=3)
ax.set_yticks(y); ax.set_yticklabels(names); ax.axvline(0.1,ls='--',color='k',lw=1)
ax.set_xlabel('Absolute standardized mean difference'); ax.legend(fontsize=8.5,loc='lower right'); f.tight_layout()
f.savefig('results/figures/fig_love_plot.png',dpi=300)
import copd_revision_analysis as A
d,_=A.load('dataset.csv'); a=d.SGRQ[d.severe==1]; b=d.SGRQ[d.severe==0]
t=stats.ttest_ind(a,b,equal_var=False); df=t.df; se=np.sqrt(a.var()/len(a)+b.var()/len(b)); diff=a.mean()-b.mean(); q=stats.t.ppf(.975,df)
S['unadjusted']={'diff':float(diff),'ci':[float(diff-q*se),float(diff+q*se)],'p':float(t.pvalue)}
json.dump(S,open('results/supplementary_results.json','w'),indent=2); print(S['unadjusted'])
