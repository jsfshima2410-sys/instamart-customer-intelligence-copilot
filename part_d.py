import json,re,time,uuid,sqlite3
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm
ROOT=Path(__file__).parent; OUT=ROOT/'outputs'; POL=ROOT/'data/policies'; STATE=ROOT/'state'; OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
for old in STATE.glob('*.json'): old.unlink()
def make_chunks(overlap=10):
 chunks=[]
 for path in sorted(POL.glob('*.txt')):
  words=path.read_text(encoding='utf-8').split(); start=0
  while start<len(words):
   end=min(start+40,len(words)); chunks.append({'source':path.name,'text':' '.join(words[start:end])})
   if end==len(words):break
   start=end-overlap
 return chunks
def new_retriever(overlap=10):
 docs=make_chunks(overlap); vec=TfidfVectorizer(); matrix=vec.fit_transform([x['text'] for x in docs])
 def retrieve(query,k=3):
  scores=cosine_similarity(vec.transform([query]),matrix)[0]; idx=np.argsort(-scores)[:k]
  return [{'source':docs[i]['source'],'score':float(scores[i]),'text':docs[i]['text']} for i in idx]
 return docs,retrieve
docs,retrieve=new_retriever(10)
def answer(question):
 found=retrieve(question,3); context='\n'.join(f"[{x['source']}]\n{x['text']}" for x in found); user=f"CONTEXT:\n{context}\nQUESTION:\n{question}"
 response=mock_llm.chat([{'role':'system','content':'POLICY ASSISTANT: answer only from the supplied context and say you could not find the information otherwise.'},{'role':'user','content':user}]); return {'answer':response.choices[0].message.content,'sources':list(dict.fromkeys(x['source'] for x in found)),'retrieval':found}
evaldf=pd.read_csv(ROOT/'data/raw/policy_eval.csv'); ev=[]
for _,r in evaldf.iterrows():
 res=answer(r.question); top=[x['source'] for x in res['retrieval']]; ev.append({'question':r.question,'expected_doc':r.expected_doc,'expected_fact':r.expected_fact,'answer':res['answer'],'doc_hit':bool(r.expected_doc in top) if pd.notna(r.expected_doc) and str(r.expected_doc) else None,'fact_hit':bool(str(r.expected_fact) in res['answer']) if pd.notna(r.expected_fact) and str(r.expected_fact) else None,'fallback':res['answer'].startswith('I could not find')})
zero_docs,zero_retrieve=new_retriever(0); zero_ev=[]
for _,r in evaldf.iterrows():
 got=zero_retrieve(r.question,3); ctx='\n'.join(f"[{x['source']}]\n{x['text']}" for x in got); u=f"CONTEXT:\n{ctx}\nQUESTION:\n{r.question}"; ans=mock_llm.chat([{'role':'system','content':'POLICY ASSISTANT: answer only from supplied context and say you could not find the information otherwise.'},{'role':'user','content':u}]).choices[0].message.content; zero_ev.append({'question':r.question,'expected_doc':r.expected_doc,'expected_fact':r.expected_fact,'answer':ans,'doc_hit':bool(r.expected_doc in [x['source'] for x in got]) if pd.notna(r.expected_doc) and str(r.expected_doc) else None,'fact_hit':bool(str(r.expected_fact) in ans) if pd.notna(r.expected_fact) and str(r.expected_fact) else None,'fallback':ans.startswith('I could not find')})
# Tool definitions are mirrored by TOOL_MAP; argument schemas are kept available to the agent.
TOOLS_SPEC=[{'name':'get_order','description':'Look up one order in the Part A database.','parameters':{'order_id':'string'}},{'name':'search_policy','description':'Retrieve the most relevant policy passages.','parameters':{'query':'string'}},{'name':'calculate_refund','description':'Calculate the amount allowed by refund rules.','parameters':{'order_amount':'number','delivery_minutes':'number'}},{'name':'issue_refund','description':'Record an approved automatic refund.','parameters':{'order_id':'string','refund_inr':'number'}},{'name':'escalate_to_human','description':'Record a refund request for human review.','parameters':{'order_id':'string','refund_inr':'number'}}]
def get_order(order_id):
 with sqlite3.connect(ROOT/'data/instamart.db') as con: row=con.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
 if not row:return {'error':'order not found'}
 return {'order_id':row[0],'status':row[1],'order_amount':float(row[2]),'delivery_minutes':float(row[3]) if row[3] is not None else None}
def search_policy(query):return {'results':retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes):return {'refund_inr':round(float(order_amount),2) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0}
def ledger(tool_name,order_id,refund_inr):
 row={'timestamp':datetime.now(timezone.utc).isoformat(),'tool_name':tool_name,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
 with (OUT/'refund_ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
 return {'refund_inr':round(float(refund_inr),2)}
def issue_refund(order_id,refund_inr):return ledger('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr):return ledger('escalate_to_human',order_id,refund_inr)
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}
class SimulatedCrash(RuntimeError):pass
def checkpoint(run_id,scenario_id,messages,tokens,steps,executed,call_count):
 data={'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens,'step_number':len(steps),'tools_executed':executed,'steps':steps,'llm_calls':call_count}
 (STATE/f'{run_id}.json').write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')
def continue_agent(data,max_iterations=6,token_limit=6000,time_limit_seconds=10,crash_after=None):
 started=time.monotonic(); run_id=data['run_id']; scenario_id=data['scenario_id']; messages=data['messages']; tokens=data['tokens_used']; steps=data['steps']; executed=data['tools_executed']; calls=data['llm_calls']; mock_llm.reset()
 while calls<max_iterations:
  if time.monotonic()-started>=time_limit_seconds:return {'status':'FALLBACK_TIMEOUT','final_answer':'FALLBACK_TIMEOUT: Please hand this request to a support agent.','steps':steps,'llm_calls':calls,'tokens_used':tokens}
  response=mock_llm.chat(messages,tools=TOOLS_SPEC); calls+=1; tokens+=response.usage.total_tokens
  if tokens>token_limit:return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':'FALLBACK_TOKEN_LIMIT: Please hand this request to a support agent.','steps':steps,'llm_calls':calls,'tokens_used':tokens}
  msg=response.choices[0].message
  if not msg.tool_calls:
   final=msg.content or ''; return {'status':final.split(':',1)[0],'final_answer':final,'steps':steps,'llm_calls':calls,'tokens_used':tokens}
  calls_payload=[{'id':tc.id,'name':tc.name,'arguments':tc.arguments} for tc in msg.tool_calls]
  messages.append({'role':'assistant','content':None,'tool_calls':calls_payload})
  for tc in msg.tool_calls:
   args=tc.arguments; observation=TOOL_MAP[tc.name](**args); step={'tool_name':tc.name,'arguments':args,'observation':observation}; steps.append(step); executed.append(tc.name)
   messages.append({'role':'tool','tool_call_id':tc.id,'name':tc.name,'content':json.dumps(observation)})
   checkpoint(run_id,scenario_id,messages,tokens,steps,executed,calls)
   if crash_after is not None and len(executed)==crash_after:raise SimulatedCrash(run_id)
 final='FALLBACK_MAX_ITERATIONS: Please hand this request to a support agent.'
 return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':final,'steps':steps,'llm_calls':calls,'tokens_used':tokens}
def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,crash_after=None):
 data={'run_id':f"{scenario_id}-{uuid.uuid4().hex[:8]}",'scenario_id':scenario_id,'messages':[{'role':'system','content':'AGENT: use tools to verify orders, retrieve policy and follow refund rules.'},{'role':'user','content':user_message}],'tokens_used':0,'steps':[],'tools_executed':[],'llm_calls':0}
 (STATE/f'{data["run_id"]}.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
 return continue_agent(data,max_iterations,token_limit,time_limit_seconds,crash_after)
def resume_agent(run_id):
 data=json.loads((STATE/f'{run_id}.json').read_text(encoding='utf-8')); return continue_agent(data)
# Reset generated ledgers for a reproducible clean run.
(OUT/'refund_ledger.jsonl').write_text('',encoding='utf-8'); (OUT/'agent_runs.jsonl').write_text('',encoding='utf-8')
sc=pd.read_csv(ROOT/'data/raw/agent_scenarios.csv'); scenario_results=[]
for _,r in sc.iterrows():
 mock_llm.reset(); result=run_agent(r.message,r.scenario_id); scenario_results.append({'scenario_id':r.scenario_id,**result})
 with (OUT/'agent_runs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'scenario_id':r.scenario_id,'status':result['status'],'final_answer':result['final_answer'],'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used'],'steps':result['steps']},default=str)+'\n')
# Requested demonstrations.
mock_llm.reset(); token_demo=run_agent(sc.iloc[0].message,'S1-token-demo',token_limit=200)
time_demo=run_agent(sc.iloc[0].message,'S1-time-demo',time_limit_seconds=0)
# Crash after second tool observation is durably saved; continue from checkpoint.
mock_llm.reset(); run_id=f"S1-resume-{uuid.uuid4().hex[:8]}"
try: run_agent(sc.iloc[0].message,'S1-resume',crash_after=2)
except SimulatedCrash as exc: run_id=str(exc)
resume_result=resume_agent(run_id)
# Replace random run id from helper with a deterministic discoverable id noted in metrics.
answerable=[x for x in ev if x['expected_doc']!='NONE']; unanswerable=[x for x in ev if x['expected_doc']=='NONE']
metrics={'chunks_total':len(docs),'chunk_counts':pd.Series([d['source'] for d in docs]).value_counts().to_dict(),'rag':{'answerable_doc_hits':sum(x['doc_hit'] for x in answerable),'answerable_fact_hits':sum(bool(x['fact_hit']) for x in answerable),'unanswerable_fallbacks':sum(x['fallback'] for x in unanswerable),'zero_overlap_doc_hits':sum(bool(x['doc_hit']) for x in zero_ev if x['doc_hit'] is not None),'zero_overlap_fact_hits':sum(bool(x['fact_hit']) for x in zero_ev if x['fact_hit'] is not None),'zero_overlap_failures':[x['question'] for x in zero_ev if x['fact_hit'] is False]},'scenarios':[{'scenario_id':x['scenario_id'],'status':x['status'],'refund_amount':next((s['observation']['refund_inr'] for s in x['steps'] if s['tool_name'] in ('issue_refund','escalate_to_human')),0),'llm_calls':x['llm_calls'],'tokens_used':x['tokens_used']} for x in scenario_results],'token_demo_status':token_demo['status'],'timeout_demo_status':time_demo['status'],'resume_tools':resume_result['steps'],'resume_status':resume_result['status'],'resume_final':resume_result['final_answer'],'resume_run_id':run_id,'s1_calls':scenario_results[0]['llm_calls'],'s7_status':scenario_results[-1]['status']}
(OUT/'policy_eval_results.json').write_text(json.dumps(ev,indent=2,default=str),encoding='utf-8'); (OUT/'part_d_metrics.json').write_text(json.dumps(metrics,indent=2,default=str),encoding='utf-8'); print(json.dumps(metrics,indent=2,default=str))