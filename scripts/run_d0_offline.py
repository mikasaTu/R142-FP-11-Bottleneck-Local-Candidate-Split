import os,glob,json,hashlib,datetime
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
RAW=ROOT/'results/stage_r/phase0r/raw'; OUT=ROOT/'stage-d/results'
for d in ['labels','signals','specificity']: (OUT/d).mkdir(parents=True,exist_ok=True)
SEED=14211; Rng=np.random.default_rng(SEED); NPERM=1000; BINS=32

def normtraj(flat,lens,idx):
 s=int(np.sum(lens[:idx])); n=int(lens[idx]); x=flat[s:s+n]
 if n<2:return np.zeros((BINS,x.shape[1]),float)
 q=np.linspace(0,n-1,BINS); lo=np.floor(q).astype(int); hi=np.minimum(lo+1,n-1); w=q-lo
 return x[lo]*(1-w[:,None])+x[hi]*w[:,None]
def fam_stat(tr,lab):
 # mean Euclidean EEF separation curve, normalized by pooled scale
 a=tr[lab]; b=tr[~lab]
 if len(a)==0 or len(b)==0:return np.zeros(BINS)
 d=np.linalg.norm(a.mean(0)-b.mean(0),axis=1)
 scale=np.median(np.linalg.norm(tr[:,1:]-tr[:,:-1],axis=2))+1e-9
 return d/scale
def onset(curve,thr):
 z=np.asarray(curve)
 hit=np.where(z>thr)[0]
 return int(hit[0]) if len(hit) else None
def detect(traj,actions):
 # all functions consume only one rollout's eef/actions; return normalized steps
 e=traj; a=actions
 spread=np.linalg.norm(e-e.mean(0),axis=1)
 d=np.gradient(spread); tc=int(np.argmax(d[1:])+1) if len(d)>2 else 0
 ad=np.linalg.norm(a,axis=1); an=int(np.argmax(np.abs(np.gradient(ad))[1:])+1) if len(ad)>2 else 0
 vel=np.linalg.norm(np.gradient(e,axis=0),axis=1); acc=np.abs(np.gradient(vel)); cp=int(np.argmax(acc[1:])+1) if len(acc)>2 else 0
 # pairwise stepwise distance to first action (within rollout)
 pd=np.linalg.norm(a-a[0],axis=1); pc=int(np.argmax(np.abs(np.gradient(pd))[1:])+1) if len(pd)>2 else 0
 return {'eef_spread':tc/BINS,'action_distance':pc/BINS,'cluster_split':pc/BINS,'proprio_changepoint':cp/BINS,'action_discontinuity':an/BINS}

def corr(x,y):
 if len(x)<3 or np.std(x)==0 or np.std(y)==0:return 0.0
 return float(np.corrcoef(x,y)[0,1])
all_labels=[]; mixed=[]; task_rows={}; allsuccess=[]
for f in sorted(RAW.glob('*.npz')):
 z=np.load(f,allow_pickle=False); task=f.stem
 lens=z['lengths']; offs=z['offsets']; succ=z['success'].astype(bool); init=z['init_state']; cand=z['candidate_id']
 # precompute normalized rollouts
 eef=np.stack([normtraj(z['eef'],lens,i) for i in range(len(lens))])
 acts=np.stack([normtraj(z['actions'],lens,i) for i in range(len(lens))])
 for st in np.unique(init):
  ids=np.where(init==st)[0]; ids=ids[np.argsort(cand[ids])]; lab=succ[ids]
  family=f'{task}:{int(st)}'; rec={'task':task,'init_state':int(st),'family':family,'n':int(len(ids)),'successes':int(lab.sum()),'failures':int((~lab).sum())}
  if len(ids)!=32 or lab.sum()<2 or (~lab).sum()<2: rec['exclusion']='NOT_MIXED'; allsuccess.append(rec) if lab.all() else None; all_labels.append(rec); continue
  tr=eef[ids]; # true separation and task-level null max statistic
  obs=fam_stat(tr,lab); mx=float(obs.max()); null=np.empty(NPERM)
  for j in range(NPERM): null[j]=fam_stat(tr,Rng.permutation(lab)).max()
  thr=float(np.quantile(null,0.95)); tc=onset(obs,thr)
  # progress onset from progress trajectories (validation-only; never detector input)
  prog=np.stack([normtraj(z['progress'][:,None],lens,i)[:,0] for i in ids])
  ps=fam_stat(prog[:,:,None],lab); pc=onset(ps,thr)
  if tc is None: reason='CENSORED'
  elif mx<=thr: reason='NO_SEPARATION'
  elif pc is not None and (pc-tc)<=0: reason='FAILURE_EVENT_NOT_FORK'
  else: reason='RETAINED'
  rec.update({'observed_max':mx,'null95':thr,'t_c':None if tc is None else tc/(BINS-1),'progress_onset':None if pc is None else pc/(BINS-1),'exclusion':None if reason=='RETAINED' else reason,'permutation_seed':SEED,'permutations':NPERM})
  all_labels.append(rec)
  if reason=='RETAINED':
   rec['detectors']={k:float(np.median([detect(eef[i],acts[i])[k] for i in ids])) for k in ['eef_spread','action_distance','cluster_split','proprio_changepoint','action_discontinuity']}
   mixed.append(rec)
# detector metrics
signals={k:[] for k in ['eef_spread','action_distance','cluster_split','proprio_changepoint','action_discontinuity']}
for r in mixed:
 tc=r['t_c']
 for k,v in r['detectors'].items(): signals[k].append((tc-v, v, tc, r['task']))
summary={}
for k,vals in signals.items():
 leads=np.array([v[0] for v in vals]); det=np.array([v[1] for v in vals]); tc=np.array([v[2] for v in vals]);
 # per-task permutation null of correlation, 1000 label permutations
 null=[]
 by={}
 for i,v in enumerate(vals): by.setdefault(v[3],[]).append(i)
 for j in range(NPERM):
  xx=[]; yy=[]
  for ids in by.values():
   p=Rng.permutation(ids); xx.extend(det[ids]); yy.extend(tc[p])
  null.append(corr(np.array(xx),np.array(yy)))
 summary[k]={'n':len(vals),'median_lead':float(np.median(leads)) if len(leads) else None,'fraction_lead_gt0':float(np.mean(leads>0)) if len(leads) else None,'corr_detect_tc':corr(det,tc),'corr_null95':float(np.quantile(null,.95)) if null else None,'leads':leads.tolist(),'detected':det.tolist(),'tc':tc.tolist(),'permutations':NPERM}
# specificity: all-success families, detector firing if detector before 0.5 normalized step
for r in all_labels:
 if r['successes']==32: r.setdefault('all_success',True)
firing={k:{'families':0,'fired':0,'firing_rate':0.0} for k in signals}
# conservative all-success firing unavailable from retained detector snapshots; recompute per all-success family using same detector
for f in sorted(RAW.glob('*.npz')):
 z=np.load(f); lens=z['lengths']; succ=z['success']; init=z['init_state']; cand=z['candidate_id']
 for st in np.unique(init):
  ids=np.where(init==st)[0]; ids=ids[np.argsort(cand[ids])]
  if len(ids)!=32 or not succ[ids].all():continue
  eef=np.stack([normtraj(z['eef'],lens,i) for i in ids]); acts=np.stack([normtraj(z['actions'],lens,i) for i in ids])
  for k in signals:
   firing[k]['families']+=1; v=np.median([detect(eef[i],acts[i])[k] for i in range(32)])
   if v<0.5:firing[k]['fired']+=1
for v in firing.values(): v['firing_rate']=v['fired']/v['families'] if v['families'] else None
labels={'freeze_commit':'PENDING_PROTOCOL_COMMIT','families_total':len(all_labels),'mixed_families':sum(r['successes']>=2 and r['failures']>=2 for r in all_labels),'retained_families':len(mixed),'excluded_counts':{},'families':all_labels,'calibration_npz_found':len(list((ROOT/'stage-s/results/calibration/B').glob('**/*.npz')))}
for r in all_labels:
 x=r.get('exclusion'); labels['excluded_counts'][x or 'RETAINED']=labels['excluded_counts'].get(x or 'RETAINED',0)+1
(OUT/'labels/labels.json').write_text(json.dumps(labels,indent=2))
(OUT/'signals/summary.json').write_text(json.dumps({'freeze_commit':'PENDING_PROTOCOL_COMMIT','signals':summary},indent=2))
(OUT/'specificity/firing_rates.json').write_text(json.dumps({'freeze_commit':'PENDING_PROTOCOL_COMMIT','signals':firing},indent=2))
print(json.dumps({'families':len(all_labels),'retained':len(mixed),'excluded':labels['excluded_counts'],'calibration_npz':labels['calibration_npz_found'],'signals':{k:{q:v[q] for q in ['median_lead','fraction_lead_gt0','corr_detect_tc','corr_null95']} for k,v in summary.items()}},indent=2))
