import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import joblib
ROOT=Path(__file__).parent; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
c=pd.read_csv(ROOT/'data/clean/customers.csv',parse_dates=['signup_date']); o=pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts'])
comp=o[o.status.eq('completed')].copy(); snap=pd.Timestamp('2025-10-01')
rfm=comp.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')); rfm['recency']=(snap-rfm.last_order).dt.days; rfm=rfm[['recency','frequency','monetary']]
scaler=StandardScaler(); z=scaler.fit_transform(rfm); km_metrics=[]
for k in range(2,9):
 model=KMeans(n_clusters=k,n_init=10,random_state=42); lab=model.fit_predict(z); km_metrics.append({'k':k,'inertia':round(model.inertia_,1),'silhouette':round(silhouette_score(z,lab),3)})
K=4; km=KMeans(n_clusters=K,n_init=10,random_state=42); rfm['segment_id']=km.fit_predict(z); prof=rfm.groupby('segment_id').agg(size=('frequency','size'),recency=('recency','mean'),frequency=('frequency','mean'),monetary=('monetary','mean'))
order=prof.monetary.sort_values().index.tolist(); names=['Occasional Essentials','At-Risk Regulars','Steady Value','Premium Champions']
name_map={sid:names[min(i,len(names)-1)] for i,sid in enumerate(order)}; premium=max(prof.index,key=lambda i:prof.loc[i,'monetary'])
rfm['segment_name']=rfm.segment_id.map(name_map); rfm['is_premium']=rfm.segment_id.eq(premium); rfm.reset_index().to_csv(OUT/'customer_segments.csv',index=False)
pca=PCA(n_components=2).fit(z)
# Label cutoff cohort, features use strictly pre-cutoff orders.
cut=pd.Timestamp('2025-09-01'); pre=o[(o.order_ts<cut)&o.status.isin(['completed','cancelled'])].copy(); precomp=pre[pre.status.eq('completed')]; cohort=precomp.customer_id.unique()
future=comp[(comp.order_ts>=cut)&(comp.order_ts<pd.Timestamp('2025-10-01'))].groupby('customer_id').size(); y=pd.Series(1,index=cohort,dtype=int); y.loc[future.index.intersection(y.index)]=0
feat=precomp.groupby('customer_id').agg(last=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean'))
feat['recency']=(cut-feat.pop('last')).dt.days
final=pre.groupby('customer_id').agg(final_orders=('order_id','nunique'),cancelled=('status',lambda x:(x=='cancelled').sum())); feat['cancellation_rate']=(final.cancelled/final.final_orders).reindex(feat.index).fillna(0)
fav=precomp.groupby('customer_id').category.agg(lambda x:x.mode().iloc[0]); feat['favorite_category']=fav
feat=feat.join(c.set_index('customer_id')[['city','age','gender','signup_date']]); feat['tenure_days']=(cut-feat.signup_date).dt.days; feat=feat.drop(columns='signup_date'); feat=feat.reindex(cohort); feat['lapsed']=y
num=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','age','tenure_days']; cat=['favorite_category','city','gender']; feat[num]=feat[num].fillna(feat[num].median()); feat[cat]=feat[cat].fillna('unknown').astype(str); X=feat[num+cat]; y=feat.lapsed.astype(int)
assert not X.isna().any().any()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
preprocess=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),cat)])
logpipe=Pipeline([('prep',preprocess),('model',LogisticRegression(max_iter=1000))]); loggrid=GridSearchCV(logpipe,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1'); loggrid.fit(Xtr,ytr)
rfpipe=Pipeline([('prep',preprocess),('model',RandomForestClassifier(n_estimators=100,random_state=42))]); rfgrid=GridSearchCV(rfpipe,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1'); rfgrid.fit(Xtr,ytr)
def score(model):
 pred=model.predict(Xte); return {'confusion_matrix':confusion_matrix(yte,pred).tolist(),'precision':round(precision_score(yte,pred,zero_division=0),3),'recall':round(recall_score(yte,pred,zero_division=0),3),'f1':round(f1_score(yte,pred,zero_division=0),3),'accuracy':round(accuracy_score(yte,pred),3)}
# Final selection: RF (higher CV/test F1 for this fixed data).
selected=rfgrid.best_estimator_; joblib.dump(selected,OUT/'lapse_model.joblib'); reloaded=joblib.load(OUT/'lapse_model.joblib'); assert np.array_equal(selected.predict(Xte),reloaded.predict(Xte))
# Training-only mean from fitted scaler.
train_freq_mean=float(selected.named_steps['prep'].named_transformers_['num'].mean_[num.index('frequency')]); full_freq_mean=float(X.frequency.mean())
# Balanced logistic experiment on original training data only.
train_bal=Xtr.copy(); y_bal=ytr.copy(); rng=np.random.RandomState(42); pos=np.flatnonzero(ytr.to_numpy()==1); neg_count=int((ytr==0).sum()); add=rng.choice(pos,size=neg_count-len(pos),replace=True); train_bal=pd.concat([Xtr,Xtr.iloc[add]],ignore_index=True); y_bal=pd.concat([ytr.reset_index(drop=True),ytr.iloc[add].reset_index(drop=True)],ignore_index=True); bal=clone(loggrid.best_estimator_); bal.fit(train_bal,y_bal)
# Premium is explicitly a post-hoc RFM label; filter test members, score with pre-cutoff model.
test_ids=Xte.index; prem_ids=set(rfm.index[rfm.is_premium]); ids=[i for i in test_ids if i in prem_ids]; probs=selected.predict_proba(Xte.loc[ids])[:,1] if ids else np.array([]); ret=pd.DataFrame({'customer_id':ids,'lapse_probability':probs,'actually_lapsed':yte.loc[ids].values}); ret=ret.sort_values('lapse_probability',ascending=False).head(10); ret.to_csv(OUT/'retention_list.csv',index=False)
# Importances and logistic coefficient signs (transformed feature names).
rf_model=selected.named_steps['model']; names_all=selected.named_steps['prep'].get_feature_names_out(); imp=sorted(zip(names_all,rf_model.feature_importances_),key=lambda x:x[1],reverse=True)[:5]
log=loggrid.best_estimator_; lognames=log.named_steps['prep'].get_feature_names_out(); coefs=log.named_steps['model'].coef_[0]; coef=sorted(zip(lognames,coefs),key=lambda x:abs(x[1]),reverse=True)[:3]
metrics={'rfm':{'customers':len(rfm),'recency_median':float(rfm.recency.median()),'recency_min':int(rfm.recency.min()),'recency_max':int(rfm.recency.max()),'frequency_mean':float(rfm.frequency.mean()),'frequency_max':int(rfm.frequency.max()),'monetary_mean':float(rfm.monetary.mean()),'monetary_max':float(rfm.monetary.max())},'kmeans':km_metrics,'segments':prof.round(2).to_dict('index'),'segment_names':name_map,'premium_segment_id':int(premium),'pca':pca.explained_variance_ratio_.round(4).tolist(),'cohort':{'n':len(feat),'lapsed':int(y.sum()),'lapsed_pct':round(y.mean()*100,2),'train':len(Xtr),'test':len(Xte),'baseline_accuracy':round(float((yte==0).mean()),3)},'scaler_frequency_mean_train':train_freq_mean,'frequency_mean_full':full_freq_mean,'logistic':{'best_params':loggrid.best_params_,'cv_f1':round(loggrid.best_score_,3),'test':score(loggrid.best_estimator_),'balanced_test':score(bal)},'forest':{'best_params':rfgrid.best_params_,'cv_f1':round(rfgrid.best_score_,3),'test':score(selected),'top_features':[(a,round(float(b),4)) for a,b in imp]},'logistic_coefficients':[(a,round(float(b),4)) for a,b in coef],'premium_retention_top10_lapsed':int(ret.actually_lapsed.sum()),'retention_n':len(ret)}
(OUT/'part_b_metrics.json').write_text(json.dumps(metrics,indent=2,default=str),encoding='utf-8'); print(json.dumps(metrics,indent=2,default=str))