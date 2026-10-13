"""Supplementary checks: (a) weighting balance + doubly adjusted matched estimate, (b) QoL model without CAT/HAD."""
import json, numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import copd_revision_analysis as A
d,_=A.load('dataset.csv'); out={}
t=d.severe.values; X=d[A.CONFOUNDERS].values.astype(float)
Xs=StandardScaler().fit_transform(X)
ps=LogisticRegression(max_iter=5000,C=1e6).fit(Xs,t).predict_proba(Xs)[:,1]
w=np.where(t==1,1/ps,1/(1-ps))
def wsmd(x,t,w):
    m1=np.average(x[t==1],weights=w[t==1]); m0=np.average(x[t==0],weights=w[t==0])
    return (m1-m0)/np.sqrt((x[t==1].var()+x[t==0].var())/2)
out['ipw_smd']={c:float(wsmd(d[c].values.astype(float),t,w)) for c in A.CONFOUNDERS}
out['ipw_max_abs_smd']=max(abs(v) for v in out['ipw_smd'].values())
ps_,pairs,_=A.ps_match(d); idx=np.array([i for p in pairs for i in p])
md=d.iloc[idx]
ols=sm.OLS(md.SGRQ.values, sm.add_constant(md[['severe']+A.CONFOUNDERS].astype(float))).fit(cov_type='cluster',cov_kwds={'groups':np.repeat(np.arange(len(pairs)),2)})
out['matched_regression']={'coef':float(ols.params['severe']),'ci':[float(x) for x in ols.conf_int().loc['severe']],'p':float(ols.pvalues['severe'])}
# QoL sensitivity without CAT, HAD
preds=[p for p in A.REG_PREDICTORS if p not in ('CAT','HAD')]
Xr=d[preds].values.astype(float); y=d.SGRQ.values; res={'mse':[],'mae':[],'r2':[]}
for fi,(tr,te) in enumerate(RepeatedKFold(n_splits=5,n_repeats=10,random_state=42).split(d)):
    pipe=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',XGBRegressor(random_state=42,n_jobs=1,subsample=0.8,verbosity=0))])
    gs=GridSearchCV(pipe,{'m__n_estimators':[100,300],'m__learning_rate':[0.05,0.1],'m__max_depth':[2,3]},cv=KFold(3,shuffle=True,random_state=42+fi),scoring='neg_mean_squared_error',n_jobs=-1).fit(Xr[tr],y[tr])
    yh=gs.predict(Xr[te]); res['mse'].append(mean_squared_error(y[te],yh)); res['mae'].append(mean_absolute_error(y[te],yh)); res['r2'].append(r2_score(y[te],yh))
out['reg_without_cat_had']={k:list(A.ci_corrected(v,80.8,20.2)) for k,v in res.items()}
out['reg_without_cat_had_predictors']=preds
json.dump(out,open('results/supplementary_results.json','w'),indent=2)
print(json.dumps(out,indent=1))
