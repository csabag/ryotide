"""Original MIT scenarios; labelled before model calls. No private items here."""
import json,sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevbench.tasks import Task,write_jsonl
rows=[]
def add(family,question,cases):
 for i,(text,paraphrase,label) in enumerate(cases):
  labels=['no','yes'] if question['type']=='noul' else ([str(j)for j in range(len(question['criteria']))] if question['type']=='score' else list(question['criteria']))
  for j,s in enumerate([text,paraphrase]):
   s=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',s)
   rows.append(Task(f'original-{family}-{i+1:02d}-{j}',family,s,question,labels,label,'public',f'original-{family}-{i+1:02d}',{'source':'JevBench original authored scenario','license':'MIT','label_basis':'Explicit rubric, reviewed before inference','exclude_reason':None}))
add('policy',{'type':'noul','instructions':'Under the stated policy, is the requested action permitted? Treat unproved required conditions as not satisfied.','criteria':{'true':'Every required condition is established and no prohibition applies.','false':'A condition is missing or a prohibition applies.'}},[
('Policy: refunds require a receipt and purchase within30 days. A customer bought12 days ago but has no receipt. Issue a refund.','A receipt AND an age of at most30 days are required. The purchase was12 days ago; proof of purchase is absent. May we refund?','no'),
('Policy: trial users may export CSV; PDF needs a paid plan. Trial user asks for CSV.','A trial account requests CSV export. Rules allow trial CSV, but reserve PDF for paid accounts.','yes'),
('Policy: visitors need an escort unless they are registered contractors. A registered contractor arrives alone. Admit them.','Escort rule exempts registered contractors. This visitor is a registered contractor without an escort. Can they enter?','yes'),
('Policy: send a reminder only when payment is overdue and no dispute is open. Payment is overdue; a dispute is open. Send reminder.','There is an open dispute on an overdue invoice. Reminders require overdue status AND absence of disputes. Is a reminder allowed?','no'),
('Policy: staff may access files in their own team. A temporary suspension overrides all access. The staff member owns the team but is suspended. Open its file.','Suspension blocks access even to own-team files. A suspended employee requests their own team file. May it be opened?','no'),
('Policy: bookings can be cancelled free until24 hours before departure, including exactly24 hours. Departure is24 hours away. Cancel free.','There are exactly24 hours until departure. Free cancellation is allowed at or before that24-hour cutoff.','yes')])
add('intent',{'type':'choice','instructions':'Select the primary requested action. A mention without a request does not establish intent.','criteria':{'cancel':'End an existing subscription','refund':'Return money already charged','status':'Learn delivery progress','change_address':'Modify a delivery address','other':'None of these actions is requested'}},[
('Do not cancel my membership. Please return the duplicate charge.','Keep my subscription active; I only want the second charge refunded.','refund'),
('Where is the parcel? Keep the delivery address as it is.','Please tell me the package progress. No address changes needed.','status'),
('The order has not shipped. Send it to my new office instead.','Please replace my shipping address with the office address before dispatch.','change_address'),
('Stop renewing my subscription after this month; I am not asking for money back.','End the membership at the next renewal. No refund requested.','cancel'),
('Your refund policy is clearer now. Thanks for explaining it.','Thank you; I understand the policy on refunds now.','other'),
('I cancelled yesterday. Was the parcel delivered yet?','The cancellation is already done. My question is whether the shipment has arrived.','status')])
add('ordinal',{'type':'score','instructions':'Rate incident impact using only reported facts. Use the highest fully supported level.','criteria':['No function impaired; cosmetic only','One user or a nonessential function impaired, with a workaround','Many users blocked from a core function, no data loss','Confirmed irreversible data loss or physical harm']},[
('The icon is misaligned. Every function works.','All features work normally; only an icon looks off. ',0),
('One user cannot download an attachment but can open it in the browser.','A single user has a download failure; viewing the attachment online still works.',1),
('All customers cannot sign in. No records are lost.','Login is unavailable to every customer, but stored data remains intact.',2),
('Backups and original customer records have been irreversibly deleted.','Customer records are permanently gone and there is no remaining backup.',3),
('A banner says possible data loss. Investigation confirms no loss and no function failure.','Despite a warning banner, checks find every function working and no missing data.',0),
('Many users cannot edit orders; read-only access still works. No data loss.','Order editing, a core function, is blocked for many users. Viewing orders is possible; nothing has been lost.',2)])
add('extraction',{'type':'choice','instructions':'Extract the final confirmed delivery method. Ignore cancelled plans and hypothetical alternatives. Choose unknown if no final method is confirmed.','criteria':{'courier':'Courier delivery','pickup':'Customer pickup','post':'Postal service','unknown':'No final confirmed method'}},[
('We considered courier, then confirmed pickup at the depot.','The final arrangement is depot pickup, replacing the earlier courier idea.','pickup'),
('If approved, we might use post. No method is booked yet.','Postal delivery is a possibility awaiting approval; nothing has been confirmed.','unknown'),
('Courier booked for Tuesday. The customer asked about pickup but did not change the booking.','Tuesday courier delivery is confirmed. A later question about pickup did not alter it.','courier'),
('Pickup was cancelled. The parcel was handed to the postal service.','After cancelling collection, we sent the package by post.','post'),
('No courier service is available. We have not decided what to do.','Courier is unavailable and no alternative has been selected.','unknown'),
('The final signed order says courier, despite an old email saying post.','An earlier postal plan was superseded by the signed courier order.','courier')])
add('adequacy',{'type':'noul','instructions':'Does the response fully satisfy the request, using the supplied reference when present?','criteria':{'true':'Correct, complete, and follows all explicit constraints','false':'Wrong, incomplete, unsupported or violates a constraint'}},[
('Request: Return only the sum of17 and25. Response:42','Task: Give just17+25. Answer:42','yes'),
('Request: Name both colors from reference. Reference: red and teal. Response:red','Reference lists red and teal. Asked to return both, the answer gives only red.','no'),
('Request: Return a JSON array containing the integer3. Response:{"value":3}','Task: Output an array with integer3. Answer is an object with a value property set to3.','no'),
('Request: Is the library open Sunday? Reference: Closed Sunday. Response:No, it is closed on Sunday.','Reference says the library is closed on Sunday. Asked whether it opens Sunday, the answer says no.','yes'),
('Request: Say exactly two words. Response:All done now','Task requires exactly two words; the response is All done now.','no'),
('Request: Which train leaves first? Reference:A at09:20, B at08:50. Response:B','Reference departures are09:20 for A and08:50 for B. The requested earliest train is answered B.','yes')])
add('routing',{'type':'choice','instructions':'Choose the specialist needed for the request. File edits with test execution use coding_agent, even if code-related.','criteria':{'math':'Self-contained calculation or proof','coding':'Self-contained code writing or explanation without repository operations','coding_agent':'Inspect/edit repository files or run tests','document':'Answer from a supplied document','tools':'Carry out an external service action','general':'None of the specialist categories'}},[
('Compute the least common multiple of12 and18.','Find LCM(12,18).','math'),
('Write a Python function that reverses a list; no files need editing.','Show standalone Python code to reverse a list.','coding'),
('Open this repository, fix the failing parser, and run its tests.','Inspect the project files, repair the parser bug, and execute the test suite.','coding_agent'),
('Read the attached contract and list its renewal dates.','Extract the renewal dates from the provided agreement.','document'),
('Use the calendar service to move my meeting to Friday.','Reschedule my meeting for Friday in the calendar app.','tools'),
('Suggest three imaginative names for a pet dragon.','Give me three creative pet-dragon names.','general')])
path=Path(__file__).resolve().parents[1]/'datasets/public/original.jsonl';path.parent.mkdir(parents=True,exist_ok=True);write_jsonl(rows,str(path));print(len(rows),path)
