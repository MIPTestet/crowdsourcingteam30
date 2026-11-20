"""Export the final dataset, provenance and crowd responses for the team repository."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from hashlib import sha256
import math

from followup import RESULTS, STATE_FILE, bbox_answers, reads
from prepare import ROOT, json, list_items, request, save

BOX_FIELDS = ('left', 'top', 'width', 'height')
FEE = Decimal('1.40')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')


def worker_alias(worker):
    return sha256(('team30-review:' + worker).encode()).hexdigest()[:16]


def pseudonymize(value):
    if isinstance(value, list):
        return [pseudonymize(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {('worker' if key == 'user_id' else key):
            (worker_alias(item) if key == 'user_id' else pseudonymize(item))
            for key, item in value.items()}


def export_pool(item):
    label, pool_id = item
    pool = request('GET', 'pools/' + pool_id)
    assignments = list_items('assignments', pool_id=pool_id, status='ACCEPTED')
    reward = sum((Decimal(str(a['reward'])) for a in assignments), Decimal('0'))
    meta = {
        'stage': label, 'pool_id': pool_id, 'project_id': pool['project_id'],
        'status': pool['status'], 'accepted_pages': len(assignments),
        'reward_rub': float(reward), 'estimated_cost_with_fee_rub': float(reward * FEE),
    }
    for key in ('defaults', 'filter', 'quality_control', 'reward_per_assignment'):
        if key in pool:
            meta[key] = pool[key]
    answers = []
    for assignment in assignments:
        alias = worker_alias(assignment['user_id'])
        tasks, solutions = assignment['tasks'], assignment['solutions']
        for index, (task, solution) in enumerate(zip(tasks, solutions, strict=True)):
            answers.append({
                'stage': label, 'pool_id': pool_id, 'assignment_id': assignment['id'],
                'worker': alias, 'position': index + 1, 'input': task['input_values'],
                'answer': solution['output_values'],
                'control': task.get('known_solutions', []),
            })
    return meta, answers


def build_targets(state):
    accepted = {r['photo_id']: r for r in read_jsonl(RESULTS / 'verified_after_final_recheck.jsonl')}
    verified_human = bbox_answers(state['human_redraw_11'])
    correction_rows = read_jsonl(RESULTS / 'golden_set_corrections.jsonl')
    corrections = {r['photo_id']: r for r in correction_rows}
    assert len(accepted) == 389 and set(corrections) == set(verified_human)
    assert not set(accepted) & set(verified_human)
    pool_id = state['human_review_11']
    votes = {}
    for assignment in list_items('assignments', pool_id=pool_id, status='ACCEPTED'):
        for task, solution in zip(assignment['tasks'], assignment['solutions'], strict=True):
            inputs = task['input_values']
            photo = inputs['photo_id']
            if photo not in verified_human:
                assert task.get('known_solutions')
                continue
            assert photo not in votes and assignment['user_id'] != verified_human[photo][2]
            assert json.loads(inputs['shown_boxes']) == (verified_human[photo][1].get('boxes') or [])
            votes[photo] = solution['output_values']['verdict']
    assert set(votes) == set(verified_human)
    for photo, (inputs, human, _) in verified_human.items():
        verdict = votes.get(photo)
        use_human = verdict == 'accept' and human['status'] == 'has_butterflies'
        current = human if use_human else corrections[photo]['after']
        origin = ('independent_human_reannotation' if use_human else 'model_assisted_adjudication')
        notes = {
            'reannotation_pool_id': state['human_redraw_11'], 'review_pool_id': pool_id,
            'verdict': verdict, 'human_candidate': human,
        }
        if verdict == 'accept' and not use_human:
            notes['retention_reason'] = 'На 667238360 видны две особи, подтверждённые соседними снимками наблюдения.'
        accepted[photo] = {
            'photo_id': photo, 'image_url': inputs['image_url'],
            'status': current['status'], 'boxes': current.get('boxes') or [],
            'source': {'type': origin}, 'final_human_review': notes,
        }
    manifest = read_jsonl(ROOT / 'data' / 'manifests' / 'main_400.jsonl')
    assert len(accepted) == len(manifest) == 400
    assert set(accepted) == {m['photo_id'] for m in manifest}
    ordered = [accepted[m['photo_id']] for m in manifest]
    targets = []
    for row in ordered:
        assert row['status'] in ('has_butterflies', 'no_butterflies')
        assert bool(row['boxes']) == (row['status'] == 'has_butterflies')
        boxes = [{k: b[k] for k in BOX_FIELDS} for b in row['boxes']]
        for box in boxes:
            left, top, width, height = box.values()
            assert all(math.isfinite(value) for value in box.values())
            assert 0 <= left < 1 and 0 <= top < 1 and width > 0 and height > 0
            assert left + width <= 1.000001 and top + height <= 1.000001
        targets.append({'photo_id': row['photo_id'], 'boxes': boxes, 'error': None})
    write_jsonl(RESULTS / 'golden_set.jsonl', targets)
    write_jsonl(RESULTS / 'golden_set_provenance.jsonl', map(pseudonymize, ordered))
    return {
        'golden_set_photos': len(targets), 'boxes': sum(len(t['boxes']) for t in targets),
        'empty': sum(not t['boxes'] for t in targets), 'final_review_received': len(votes),
        'final_review_verdicts': dict(Counter(votes.values())),
        'human_replacements': sum(accepted[p]['source']['type'] == 'independent_human_reannotation'
                                  for p in verified_human),
    }


def main():
    state = reads(STATE_FILE)
    stages = {
        'bbox_training': '6867719', 'bbox_exam': '6867720', 'bbox': '6867722',
        'review_training': '6872482', 'review_exam': '6872483', 'review_pilot': '6872504',
        'review': '6872484', 'bbox_extra_exam': '6874967', 'reannotation_131': '6874984',
        'review_131': '6875096', **state,
    }
    summary = build_targets(state)
    summary['model_assisted_retained'] = 11 - summary['human_replacements']
    with ThreadPoolExecutor(max_workers=5) as workers:
        exports = list(workers.map(export_pool, stages.items()))
    write_jsonl(RESULTS / 'stage_answers.jsonl', (row for _, rows in exports for row in rows))
    pools = [p for p, _ in exports]
    total = sum((Decimal(str(p['estimated_cost_with_fee_rub'])) for p in pools), Decimal('0'))
    summary['pools_estimated_cost_with_fee_rub'] = float(total)
    summary['balance_rub'] = request('GET', 'requester')['balance']
    save(RESULTS / 'pools_public.json', pools)
    save(RESULTS / 'delivery_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
