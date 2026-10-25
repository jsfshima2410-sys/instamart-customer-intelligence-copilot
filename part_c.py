import json,re,random,time
from pathlib import Path
import pandas as pd
import mock_llm
random.seed(42)
ROOT=Path(__file__).parent; RAW=ROOT/'data/raw'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
MAX_INPUT_TOKENS=400; BASE_DELAY=.05; ALLOWED={'billing','delivery','technical','fraud'}
PATTERNS=[('email',re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')) ,('phone',re.compile(r'\b[6-9]\d{9}\b')),('order_id',re.compile(r'ORD-\d{6}'))]
ZERO='CLASSIFY each support ticket. Allowed categories: billing, delivery, technical, fraud. Reply with JSON only in the form {"category": "...", "reason": "..."}.'
FEW='CLASSIFY each support ticket. Allowed categories: billing, delivery, technical, fraud. Reply with JSON only in the form {"category": "...", "reason": "..."}.\nEXAMPLES:\nTicket: I was charged twice. -> {"category":"billing","reason":"duplicate charge"}\nTicket: Someone hacked my account without permission. -> {"category":"fraud","reason":"unauthorized access"}'
def redact(text):
 counts={}
 for name,pat in PATTERNS: text,n=pat.subn({'email':'[EMAIL]','phone':'[PHONE]','order_id':'[ORDER_ID]'}[name],text); counts[name]=n
 return text,counts
def classify_reply(text):
 try:
  obj=json.loads(text); cat=obj.get('category')
  if cat in ALLOWED:return cat,'classified_json'
 except (json.JSONDecodeError,AttributeError,TypeError): pass
 words=re.findall(r'[a-z]+',(text or '').lower()); matches=[x for x in ALLOWED if x in words]
 if len(matches)==1:return matches[0],'classified_keyword'
 return '', 'flagged_invalid_output'
def run(prompt,budget=None,log_retries=False):
 rows=[]; total=0; retry_log=[]; mock_llm.reset(); df=pd.read_csv(RAW/'tickets.csv',keep_default_na=False)
 red_tot={'email':0,'phone':0,'order_id':0}
 for _,r in df.iterrows():
  txt=str(r.text); safe,cnt=redact(txt)
  for k,v in cnt.items():red_tot[k]+=v
  for _,pat in PATTERNS: assert not pat.search(safe)
  rec={'ticket_id':r.ticket_id,'customer_id':r.customer_id,'status':'','category':'','tokens_used':0,'true_category':r.true_category}
  if not txt.strip(): rec['status']='skipped_empty'
  elif mock_llm.est_tokens(txt)>MAX_INPUT_TOKENS:rec['status']='flagged_long'
  elif budget is not None and total>=budget:rec['status']='budget_exhausted'
  else:
   messages=[{'role':'system','content':prompt},{'role':'user','content':safe}]
   assert messages[-1]['content']==safe
   tries=0
   while True:
    try:
     response=mock_llm.chat(messages); break
    except mock_llm.TransientLLMError:
     if tries>=3: raise
     wait=BASE_DELAY*(2**tries)*random.uniform(.9,1.15); line=f"retry ticket={r.ticket_id} retry={tries} wait={wait:.4f}s"; print(line); retry_log.append(line); time.sleep(wait); tries+=1
   used=response.usage.total_tokens; total+=used; rec['tokens_used']=used
   rec['category'],rec['status']=classify_reply(response.choices[0].message.content)
  rows.append(rec)
 result=pd.DataFrame(rows); classified=result.status.isin(['classified_json','classified_keyword']); accuracy=float((result.loc[classified,'category']==result.loc[classified,'true_category']).mean())
 summary={'rows':len(result),'classified':int(classified.sum()),'accuracy':round(accuracy,4),'tokens':int(total),'tokens_per_classified':round(total/max(1,int(classified.sum())),2),'status_counts':result.status.value_counts().to_dict(),'redactions':red_tot,'redactions_total':sum(red_tot.values()),'retry_count':len(retry_log)}
 summary['retry_log']=retry_log
 return result,summary
zero,zs=run(ZERO); few,fs=run(FEW,budget=100000)
# confusion matrix for zero shot, only accepted classifications
from sklearn.metrics import confusion_matrix
labels=['billing','delivery','technical','fraud']; zc=zero[zero.status.isin(['classified_json','classified_keyword'])]; cm=pd.crosstab(zc.true_category,zc.category).reindex(index=labels,columns=labels,fill_value=0)
# Budget demo uses zero-shot with the requested 6000 token cap.
bud,bs=run(ZERO,budget=6000); bud.to_csv(OUT/'triage_budget_demo.csv',index=False)
segments=pd.read_csv(OUT/'customer_segments.csv').set_index('customer_id'); out=[]
for _,r in few.iterrows():
 seg=segments.loc[r.customer_id] if r.customer_id in segments.index else None; seg_name=seg.segment_name if seg is not None else ''
 if r.status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}:route,priority='human_review','NORMAL'
 elif r.category=='fraud':route,priority='security_team','HIGH'
 elif seg is not None and bool(seg.is_premium):route,priority='premium_desk','HIGH'
 else:route,priority='support_queue','NORMAL'
 out.append({'ticket_id':r.ticket_id,'customer_id':r.customer_id,'status':r.status,'category':r.category,'segment_name':seg_name,'priority':priority,'route':route,'tokens_used':int(r.tokens_used)})
pd.DataFrame(out).to_csv(OUT/'triage_results.csv',index=False)
zero.to_csv(OUT/'triage_zero_shot.csv',index=False); few.to_csv(OUT/'triage_few_shot.csv',index=False)
(OUT/'retry_log.txt').write_text('\n'.join(zs['retry_log']+fs['retry_log'])+'\n',encoding='utf-8')
metrics={'zero_shot':{k:v for k,v in zs.items() if k!='retry_log'},'few_shot':{k:v for k,v in fs.items() if k!='retry_log'},'budget_demo':{k:v for k,v in bs.items() if k!='retry_log'},'zero_confusion':cm.T.to_dict(),'zero_confusion_labels':labels}
(OUT/'part_c_metrics.json').write_text(json.dumps(metrics,indent=2,default=str),encoding='utf-8'); print(json.dumps(metrics,indent=2,default=str))