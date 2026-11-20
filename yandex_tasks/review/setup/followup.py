"""Stage and open the Team 30 follow-up pools using the completed assignments."""
import argparse
from collections import defaultdict
from decimal import Decimal

from prepare import ROOT, REVIEW, json, list_items, make_config, pool, read_tsv, request, save, suite, task

RESULTS = ROOT / 'results' / 'review'
STATE_FILE = RESULTS / 'followup_pools.json'
REVIEW_PROJECT = '11957'
BBOX_PROJECT = '11838'
BBOX_SKILL = '21502'
REVIEW_SKILL = '21829'
TRAINING = '6872482'


def reads(path):
    return json.loads(path.read_text())


def bbox_answers(pool_id):
    rows = {}
    for assignment in list_items('assignments', pool_id=pool_id, status='ACCEPTED'):
        for question, answer in zip(assignment['tasks'], assignment['solutions'], strict=True):
            fields = question['input_values']
            photo = int(fields['photo_id'])
            if photo in rows:
                raise ValueError(f'Duplicate Bbox submission for {photo}')
            rows[photo] = fields, answer['output_values'], assignment['user_id']
    return rows


def previous_reviewers(pool_id):
    reviewers = defaultdict(set)
    for assignment in list_items('assignments', pool_id=pool_id, status='ACCEPTED'):
        for question in assignment['tasks']:
            reviewers[int(question['input_values']['photo_id'])].add(assignment['user_id'])
    return reviewers


def qualified(skill, threshold):
    return {x['user_id'] for x in list_items('user-skills', skill_id=skill) if x['value'] >= threshold}


def review_question(row):
    fields, output, _ = row
    return task({
        'INPUT:photo_id': fields['photo_id'],
        'INPUT:image_url': fields['image_url'],
        'INPUT:shown_status': output['status'],
        'INPUT:shown_boxes': json.dumps(output.get('boxes') or []),
    })


def arrange(ids, questions, excluded, audience, gold=(), allow_incomplete=False):
    pages = []
    for index, start in enumerate(range(0, len(ids), 5)):
        photo_ids = ids[start:start + 5]
        if allow_incomplete and not all(photo in questions for photo in photo_ids):
            continue
        tasks = [questions[photo] for photo in photo_ids]
        people = set().union(*(excluded[photo] for photo in photo_ids))
        if not audience - people:
            raise ValueError(f'No eligible workers for page {index + 1}')
        if gold:
            tasks.insert(index % (len(tasks) + 1), gold[index % len(gold)])
        pages.append((tasks, people))
    return pages


def fingerprint(tasks):
    return tuple(
        (item['input_values'], item.get('known_solutions')) for item in tasks
    )


def upload_and_open(state, key, configuration, pages):
    if key not in state:
        state[key] = request('POST', 'pools', configuration)['id']
        save(STATE_FILE, state)
    pid = state[key]
    print(key, pid, flush=True)
    wanted = {tuple(t['input_values']['photo_id'] for t in tasks): (tasks, people)
              for tasks, people in pages}
    actual = list_items('task-suites', pool_id=pid)
    uploaded = {tuple(t['input_values']['photo_id'] for t in page['tasks']) for page in actual}
    pending = [suite(pid, tasks, 1, people) for signature, (tasks, people) in wanted.items()
               if signature not in uploaded]
    if pending:
        request('POST', 'task-suites', pending)
    actual = list_items('task-suites', pool_id=pid)
    assert len(actual) == len(wanted)
    for page in actual:
        ids = tuple(t['input_values']['photo_id'] for t in page['tasks'])
        tasks, people = wanted[ids]
        assert page['overlap'] == 1 and not page['infinite_overlap']
        assert set(page['unavailable_for']) == people
        assert fingerprint(page['tasks']) == fingerprint(tasks)
    path = 'pools/' + pid
    settings = request('GET', path)
    assert settings['project_id'] == configuration['project_id']
    assert settings['reward_per_assignment'] == configuration['reward_per_assignment']
    assert settings['filter'] == configuration['filter']
    if settings['status'] != 'OPEN' and (pending or not settings.get('last_started')):
        op = request('POST', path + '/open')
        print('open operation', op.get('status'), flush=True)
    settings = request('GET', path)
    assert settings['status'] == 'OPEN' or settings.get('last_started')
    print('verified', pid, settings['status'], len(actual), flush=True)
    return pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-redraw', action='store_true')
    args = parser.parse_args()
    state = reads(STATE_FILE) if STATE_FILE.exists() else {}
    review_source = reads(RESULTS / 'bbox_relabel_review_pool.json')
    rows = reads(RESULTS / 'pending_after_recheck.json')
    split = [r['photo_id'] for r in rows if r['review']['verdicts'].count('accept') == 1]
    redraw = [r['photo_id'] for r in rows if 'accept' not in r['review']['verdicts']]
    assert len(split) == 56 and len(redraw) == 20 and not set(split) & set(redraw)
    assert all(request('GET', 'projects/' + p)['owner']['myself'] for p in (REVIEW_PROJECT, BBOX_PROJECT))
    gold = [task(r) for r in read_tsv(REVIEW / 'tasks_honeypots.tsv')]
    assert not (set(split) | set(redraw)) & {t['input_values']['photo_id'] for t in gold}
    _, review = make_config(REVIEW_PROJECT, TRAINING, REVIEW_SKILL)
    box_source = request('GET', 'pools/6867722')
    review_audience = qualified(REVIEW_SKILL, 75)
    if args.review_redraw:
        new = {p: a for p, a in bbox_answers(state['redraw_20']).items()
               if a[1]['status'] != 'load_error'}
        assert set(new) <= set(redraw)
        print(f'Redraw photos ready for review: {len(new)}/{len(redraw)}', flush=True)
        pages = arrange(redraw, {p: review_question(a) for p, a in new.items()},
                        {p: {a[2]} for p, a in new.items()}, review_audience, gold,
                        allow_incomplete=True)
        if not pages:
            return
        config = pool(REVIEW_PROJECT, 'проверка новых рамок на 20 фото', 0.20, 1, 600,
                      review['filter']['and'], review['quality_control'])
        config['private_comment'] = 'Независимая проверка рамок из пула ' + state['redraw_20']
        upload_and_open(state, 'redraw_review_20', config, pages)
        return

    assert Decimal(str(request('GET', 'requester')['balance'])) >= Decimal('6.44')
    source = bbox_answers(review_source['source_pool'])
    seen = previous_reviewers(review_source['pool_id'])
    assert all(len(seen[p]) == 2 and source[p][2] not in seen[p] for p in split + redraw)
    split_pages = arrange(split, {p: review_question(source[p]) for p in split},
                          {p: seen[p] | {source[p][2]} for p in split}, review_audience, gold)
    split_pool = pool(REVIEW_PROJECT, 'третий голос на 56 фото', 0.20, 1, 600,
                      review['filter']['and'], review['quality_control'])
    split_pool['private_comment'] = 'Дополнительная проверка после двух разных вердиктов в пуле ' + review_source['pool_id']

    old = {item['user_id'] for item in list_items('assignments', pool_id='6867722', status='ACCEPTED')}
    redraw_audience = qualified(BBOX_SKILL, 80)
    redraw_pages = arrange(redraw, {p: {'input_values': source[p][0]} for p in redraw},
                           {p: old | {source[p][2]} for p in redraw}, redraw_audience)
    redraw_pool = pool(BBOX_PROJECT, 'новые рамки на 20 фото', 0.35, 1, 900,
                       box_source['filter']['and'], {'configs': []})
    redraw_pool['private_comment'] = 'Перерисовка 20 фото, которые отклонили оба ревьюера. Прежние авторы исключены из выдачи.'
    upload_and_open(state, 'redraw_20', redraw_pool, redraw_pages)
    upload_and_open(state, 'third_review_56', split_pool, split_pages)


if __name__ == '__main__':
    main()
