"""Read accepted Review answers and export verified rows plus an adjudication queue."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path

from prepare import HERE, ROOT, VERDICTS, list_items, prepare, save, task


def read_votes(assignments, photo_ids, excluded):
    votes = defaultdict(dict)
    controls = defaultdict(list)
    for assignment in assignments:
        user = assignment['user_id']
        if user in excluded:
            raise ValueError('First-stage annotator appeared in Review')
        tasks = assignment['tasks']
        solutions = assignment['solutions']
        if len(tasks) != len(solutions):
            raise ValueError(f'Mismatched tasks and solutions in {assignment["id"]}')
        for task, solution in zip(tasks, solutions):
            photo = int(task['input_values']['photo_id'])
            verdict = solution['output_values']['verdict']
            if verdict not in VERDICTS:
                raise ValueError(f'Unknown verdict for {photo}: {verdict}')
            if photo not in photo_ids:
                controls[photo].append(verdict)
                continue
            if user in votes[photo]:
                raise ValueError(f'Duplicate reviewer for photo {photo}')
            votes[photo][user] = verdict
    return votes, controls


def partition(rows, votes):
    verified, pending = [], []
    for row in rows:
        photo = int(row['photo_id'])
        counts = Counter(votes.get(photo, {}).values())
        if counts['accept'] >= 2 and sum(counts.values()) == counts['accept']:
            verified.append({k: row[k] for k in ('photo_id','image_url','status','boxes','source')})
        else:
            reason = 'needs_annotation' if row['status']=='load_error' else 'needs_review'
            if sum(counts.values()) >= 2:
                reason = 'needs_adjudication'
            pending.append({'photo_id':photo,'image_url':row['image_url'],'candidate_boxes':row['boxes'],
                            'reason':reason,'votes':dict(counts)})
    return verified, pending


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--corrections',type=Path,help='JSONL of manually resolved pending photos')
    args=parser.parse_args()
    state=json.loads((HERE/'remote.json').read_text())
    _, _, _, gold, _, excluded=prepare()
    rows=[json.loads(line) for line in (ROOT/'results/bbox/annotations_400.jsonl').read_text().splitlines()]
    ids={int(row['photo_id']) for row in rows}
    excluded |= {a['user_id'] for a in list_items('assignments',pool_id='6867722',status='ACCEPTED')}
    assignments=list_items('assignments',pool_id=state['main'],status='ACCEPTED')
    votes, controls=read_votes(assignments,ids,excluded)
    gold_labels={t['input_values']['photo_id']:t['known_solutions'][0]['output_values']['verdict'] for t in gold}
    unknown=set(controls)-set(gold_labels)
    if unknown:
        raise ValueError(f'Unexpected control IDs: {sorted(unknown)}')
    verified, pending=partition(rows,votes)
    corrections=[]
    if args.corrections:
        by_id={int(row['photo_id']):row for row in rows}
        pending_ids={row['photo_id'] for row in pending}
        resolved=set()
        with args.corrections.open() as f:
            for record in map(json.loads,f):
                photo=int(record['photo_id'])
                if photo not in pending_ids or photo in resolved:
                    raise ValueError(f'Photo {photo} is not pending or appears twice')
                status=record['status']
                boxes=record['boxes']
                reviewer=record['reviewer'].strip()
                reason=record['reason'].strip()
                if not reviewer or not reason:
                    raise ValueError(f'Missing reviewer or correction reason for photo {photo}')
                original=by_id[photo]
                task({'INPUT:photo_id':photo,'INPUT:image_url':original['image_url'],
                      'INPUT:shown_status':status,'INPUT:shown_boxes':json.dumps(boxes)})
                verified.append({'photo_id':original['photo_id'],'image_url':original['image_url'],
                                 'status':status,'boxes':boxes,
                                 'source':{'type':'manual_adjudication','reviewer':reviewer}})
                corrections.append({'photo_id':photo,'reviewer':reviewer,'reason':reason,
                                    'before':{'status':original['status'],'boxes':original['boxes']},
                                    'after':{'status':status,'boxes':boxes},
                                    'votes':list(votes.get(photo,{}).values()),
                                    'model_assisted':bool(record.get('model_assisted',False))})
                resolved.add(photo)
        pending=[row for row in pending if row['photo_id'] not in resolved]
        verified.sort(key=lambda row:int(row['photo_id']))
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'verified.jsonl').open('w') as f:
        for row in verified:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    save(args.out/'pending.json',pending)
    save(args.out/'review_verdicts.json',{photo:list(v.values()) for photo,v in votes.items()})
    save(args.out/'correction_log.json',corrections)
    summary={'verified':len(verified),'pending':len(pending),'target':len(rows),
             'complete':not pending,'gold_responses':sum(map(len,controls.values())),
             'gold_correct':sum(v==gold_labels[p] for p,vs in controls.items() for v in vs)}
    save(args.out/'summary.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':
    main()
