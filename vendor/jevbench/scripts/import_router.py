"""Import local router tasks without redistributing third-party source data.
Run: python scripts/import_router.py /path/to/auto-model-router-job /path/outside/repo
"""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevbench.tasks import Task,write_jsonl
root=Path(sys.argv[1]).resolve();dest=Path(sys.argv[2]).resolve();repo=Path(__file__).resolve().parents[1]
if dest.is_relative_to(repo):raise SystemExit('Imported third-party text must remain outside public repo')
dest.mkdir(exist_ok=True,parents=True)
p=root/'repo/experiments/tasks/tasks.jsonl';source=[json.loads(l)for l in p.read_text().splitlines()];by={t['id']:t for t in source}
q={'type':'choice','instructions':'Classify the requested work. Repository work requiring file inspection or test execution is agentic; standalone code is coding. Classify the request, do not execute it.','criteria':{'general':'Other requests','coding':'Self-contained code writing or explanation','agentic':'Multi-step repository work with file access or test execution','math':'Calculation or proof','knowledge':'Factual knowledge','long_context':'Read and reason over a supplied long document','tool_use':'Operate external services/tools, not primarily code','design':'Create or critique a user interface','summarisation':'Condense supplied text'}}
def request(t):
 if t['grader']in('agent_humaneval','agent_stdio'):
  text=t['instruction']
  for f in ['PROBLEM.md','solution.py']:
   if f in t.get('files',{}):text+='\n\n'+f+':\n'+t['files'][f];break
  return {'request':text,'environment':'Coding agent with read_file, write_file, run_tests.'}
 text=t['prompt']
 if len(text)>5000:text=text[:2000]+f'\n...[{len(text)//4} estimated tokens of document elided]...\n'+text[-2500:]
 return {'request':text,'environment':'Order-management tools are available.' if t['category']=='tool_use' else ''}
rows=[Task('router-'+t['id'],'routing',request(t),q,list(q['criteria']),t['category'],'public',None,{'source':'https://github.com/fstandhartinger/auto-model-router','source_id':t['id'],'license':'Upstream dataset terms apply; source text not redistributed','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'notes':'Original 78-task request rendering: head/tail for documents; exact nine-category accuracy, no agentic/coding equivalence.'})for t in source]
write_jsonl(rows,str(dest/'router.jsonl'))
# Reuse known grader labels with exact saved answers. Restrict to self-contained
# math: contradictory code specifications and unseen document context cannot
# silently become judge ground truth. Full answer, no truncation on our side.
p=root/'runs/verify-20260918/answers.jsonl';ans=[json.loads(l)for l in p.read_text().splitlines()]
jq={'type':'noul','instructions':'Does the response fully and correctly satisfy the request? Check the mathematics and every explicit requirement.','criteria':{'true':'Correct and complete','false':'Wrong, incomplete, or noncompliant'}}
judges=[];excluded=[]
for i,a in enumerate(ans):
 t=by.get(a['task']);reason=None
 if not t:reason='Task source missing'
 elif a['category']!='math':reason='Conservative v1 exclusion: coding spec/test disagreement risk or absent document context'
 elif not a.get('answer','').strip():reason='Empty stored answer; not a useful adequacy example'
 elif a.get('truncated'):reason='Generation truncated; exclude from semantic-adequacy pilot'
 if reason:excluded.append({'task':a['task'],'model':a['model'],'reason':reason});continue
 judges.append(Task('legacy-answer-'+str(i),'adequacy',{'request':t['prompt'],'response':a['answer']},jq,['no','yes'],'yes' if a['correct']else'no','public',None,{'source':'https://github.com/fstandhartinger/auto-model-router','source_id':a['task']+':'+a['model'],'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'license':'Upstream terms; full source not redistributed','label_basis':'Saved deterministic grader on this exact answer, not labels from a different generation'}))
write_jsonl(judges,str(dest/'judge.jsonl'));(dest/'import-exclusions.json').write_text(json.dumps(excluded,indent=2));print({'routing':len(rows),'adequacy':len(judges),'excluded_answers':len(excluded)})
