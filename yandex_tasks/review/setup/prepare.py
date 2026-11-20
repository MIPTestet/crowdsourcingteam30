"""Prepare closed Team 30 pools. Never starts pools or submits to moderation."""
import argparse
import csv
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent
ROOT = HERE.parents[2]
VERDICTS = ['accept', 'reject_missed', 'reject_extra', 'reject_inaccurate', 'load_error']
BASE = 'https://tasks.yandex.ru/api/v1/'
RU = {'category': 'profile', 'key': 'languages', 'operator': 'IN', 'value': 'RU'}
NAME = 'team30_review_v1'


def request(method, path, body=None):
    token = os.environ['TEAM30_TASKS_TOKEN']
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = Request(BASE + path, data=data, method=method,
                  headers={'Authorization': 'OAuth '+token, 'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=30) as res:
            return json.load(res)
    except HTTPError as err:
        raise RuntimeError(f'{method} {path}: {err.code} {err.read().decode().replace(token, "[redacted]")}') from None


def list_items(path, **filters):
    items = []
    while True:
        page = request('GET', path+'?'+urlencode(dict(filters, limit=100, sort='id')))
        items.extend(page['items'])
        if not page['has_more']:
            return items
        filters['id_gt'] = page['items'][-1]['id']


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def read_tsv(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def task(row):
    boxes = json.loads(row['INPUT:shown_boxes'])
    for box in boxes:
        x,y,w,h = (box[k] for k in ('left','top','width','height'))
        assert box['shape']=='rectangle' and 0<=x<1 and 0<=y<1 and w>0 and h>0
        assert x+w<=1.000001 and y+h<=1.000001
    status = row['INPUT:shown_status']
    assert status in ('has_butterflies','no_butterflies') and bool(boxes)==(status=='has_butterflies')
    result = {'input_values':{'photo_id': int(row['INPUT:photo_id']), 'image_url': row['INPUT:image_url'],
                              'shown_boxes': json.dumps(boxes,separators=(',',':'))}}
    if 'GOLDEN:verdict' in row:
        verdict = row['GOLDEN:verdict']
        assert verdict in VERDICTS[:-1]
        result['known_solutions'] = [{'output_values':{'verdict': verdict}, 'correctness_weight': 1}]
    if 'HINT:text' in row:
        assert row['HINT:text'].strip()
        result['message_on_unknown_solution'] = row['HINT:text']
    return result


def prepare():
    candidates = read_tsv(ROOT/'results/bbox/review_candidates.tsv')
    raw = [json.loads(line) for line in (ROOT/'results/bbox/annotations_400.jsonl').read_text().splitlines()]
    missing = [r for r in raw if r['status']=='load_error']
    main = [task(r) for r in candidates]
    train, exam, gold = ([task(r) for r in read_tsv(REVIEW/f'tasks_{s}.tsv')]
                        for s in ('train','exam','honeypots'))
    ids = [t['input_values']['photo_id'] for t in main+train+exam+gold]
    assert len(ids)==len(set(ids))==421
    assert [len(main),len(train),len(exam),len(gold),len(missing)] == [399,6,8,8,1]
    source_ids = {int(r['photo_id']) for r in raw}
    assert source_ids == set(ids[:399]) | {int(missing[0]['photo_id'])}
    authors = {r['INPUT:bbox_user_id'] for r in candidates}
    assert '' not in authors
    return main, train, exam, gold, missing[0], authors


def project():
    field = lambda typ: {'type':typ,'required':True,'hidden':False}
    return {'public_name':'Проверьте рамки вокруг бабочек на фото',
            'public_description':'Посмотрите на снимок и укажите, правильно ли отмечены все бабочки.',
            'public_instructions':(HERE/'instructions.html').read_text(),
            'private_comment':NAME+' | Этап 2: независимая проверка Bbox-пула 6867722.',
            'assignments_issuing_type':'AUTOMATED',
            'assignments_automerge_enabled':False,
            'max_active_assignments_count':1,
            'localization_config':{'default_language':'RU'},
            'task_spec':{
                'input_spec': {'photo_id':field('integer'),
                               'image_url':field('url'),'shown_boxes':field('string')},
                'output_spec':{'verdict':dict(field('string'),allowed_values=VERDICTS)},
                'view_spec':{'type':'tb',
                             'config':json.dumps(json.loads((HERE/'interface.template-builder.json').read_text()),
                                                 ensure_ascii=False,separators=(',',':')),
                             'lock':json.loads((HERE/'interface.template-builder.lock.json').read_text()),
                             'inferDataSpec':False}}}


def pool(project_id, label, reward, overlap, seconds, filters, qc):
    return {'project_id':project_id,'private_name':NAME+' — '+label,'reward_per_assignment':reward,
            'assignment_max_duration_seconds':seconds,'auto_accept_solutions':True,
            'may_contain_adult_content':False,'auto_close_after_complete_delay_seconds':0,
            'will_expire':'2026-12-01T20:59:59','filter':{'and':filters},'quality_control':qc,
            'defaults':{'default_overlap_for_new_task_suites':overlap},
            'assignments_issuing_config':{'issue_task_suites_in_creation_order':False}}


def suite(pool_id, tasks, overlap, exclusions=()):
    return {'pool_id':pool_id,'tasks':tasks,'overlap':overlap,'infinite_overlap':False,
            'unavailable_for':sorted(exclusions)}


def make_config(project_id, training_id, skill_id):
    exam_qc = {'training_requirement':{'training_pool_id':training_id,'training_passing_skill_value': 80},
               'configs':[{'collector_config':{'type':'GOLDEN_SET','parameters':{'history_size':8}},
                           'rules':[{'conditions':[{'key':'golden_set_answers_count','operator':'EQ','value':8}],
                                     'action':{'type':'SET_SKILL_FROM_OUTPUT_FIELD','parameters':{
                                         'skill_id':skill_id,'from_field':'golden_set_correct_answers_rate'}}}]}]}
    main_qc = {'configs':[{'collector_config':{'type':'GOLDEN_SET','parameters':{'history_size':8}},
                           'rules':[{'conditions':[{'key':'golden_set_answers_count','operator':'GTE','value':5},
                                                  {'key':'golden_set_correct_answers_rate','operator':'LT','value':75}],
                                     'action':{'type':'RESTRICTION_V2','parameters':{'scope':'POOL',
                                                                'duration_unit':'PERMANENT',
                                                                'private_comment':'Менее 75% на 5+ контролях Review'}}}]}]}
    gate = {'category':'skill','key':skill_id,'operator':'GTE','value':75}
    return (pool(project_id,'экзамен',0.1,24,600,[RU],exam_qc),
            pool(project_id,'проверка',1.5,2,600,[RU, gate],main_qc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', action='store_true')
    args=parser.parse_args()
    candidates, train, exam, gold, missing, authors = prepare()
    estimate = {'review':336,'exam':3.36,'training':0,'missing_bbox':8.4,'pilot':12.6,'total_with_fee':360.36}
    print('399 review tasks, 80 pages with 2 votes, 24 exam completions;',estimate,flush=True)
    if not args.stage:
        return
    state_path = HERE/'remote.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    original=request('GET','projects/11838')
    source_pool=request('GET','pools/6867722')
    assert original['owner']['myself'] and source_pool['project_id']=='11838'
    account=request('GET','requester')
    assert account['balance']>=estimate['total_with_fee']
    authors |= {a['user_id'] for a in list_items('assignments',pool_id='6867722',status='ACCEPTED')}
    def create(key,path,payload):
        if key not in state:
            state[key] = request('POST',path,payload)['id']
            save(state_path,state)
        print(key,state[key],flush=True)
        return state[key]
    project_id = create('project','projects',project())
    training_id = create('training','trainings',{'project_id':project_id,'private_name':NAME+' — обучение',
            'inherited_instructions':True,'assignment_max_duration_seconds':600,'mix_tasks_in_creation_order':True,
            'shuffle_tasks_in_task_suite':False,'training_tasks_in_task_suite_count':3,
            'task_suites_required_to_pass':2,'retry_training_after_days':1,
            'may_contain_adult_content':False,'public_name':'Научитесь проверять рамки вокруг бабочек',
            'public_description':'Шесть примеров с объяснениями.'})
    skill = create('exam_skill','skills',{'name':NAME+'_exam','hidden':True,
                                          'private_comment':'Доля правильных вердиктов на 8 заданиях экзамена Review.'})
    exam_config, main_config = make_config(project_id,training_id,skill)
    exam_id=create('exam','pools',exam_config)
    pilot_qc = {'configs':[{'collector_config':{'type':'ANSWER_COUNT','parameters':{}},
                            'rules':[{'conditions':[{'key':'assignments_accepted_count','operator':'GTE','value':2}],
                                      'action':{'type':'RESTRICTION_V2','parameters':{'scope':'POOL',
                                                                'duration_unit':'PERMANENT',
                                                                'private_comment':'Лимит: 2 принятые страницы Review-пилота на исполнителя.'}}}]}]}
    pilot_config=pool(project_id,'пилот',1.5,2,600,main_config['filter']['and'],pilot_qc)
    pilot_id=create('pilot','pools',pilot_config)
    main_id=create('main','pools',main_config)
    bbox=pool('11838','доразметка одного фото',2,3,600,source_pool['filter']['and'],{'configs':[]})
    bbox_id=create('missing_bbox','pools',bbox)

    def upload_tasks(key, pool_id, rows):
        existing = list_items('tasks',pool_id=pool_id)
        uploaded_ids = {t['input_values']['photo_id'] for t in existing}
        pending = [dict(t,pool_id=pool_id,infinite_overlap=True) for t in rows
                   if t['input_values']['photo_id'] not in uploaded_ids]
        if pending:
            request('POST','tasks?allow_defaults=true',pending)
        print(key, len(rows),flush=True)

    def upload_pages(key,pool_id,pages):
        existing=list_items('task-suites',pool_id=pool_id)
        signature=lambda page:tuple(t['input_values']['photo_id'] for t in page['tasks'])
        uploaded={signature(page) for page in existing}
        pending=[p for p in pages if signature(p) not in uploaded]
        for start in range(0,len(pending),20):
            request('POST','task-suites',pending[start:start+20])
        saved=list_items('task-suites',pool_id=pool_id)
        assert len(saved)==len(pages)
        expected={signature(page):page for page in pages}
        for page in saved:
            spec=expected[signature(page)]
            assert page['overlap']==spec['overlap'] and set(page['unavailable_for'])==set(spec['unavailable_for'])
        print(key,len(saved),flush=True)

    upload_tasks('training_tasks',training_id,train)
    upload_pages('exam_pages',exam_id,[suite(exam_id,exam,24,authors)])
    pages=[]
    for page_no,start in enumerate(range(0,len(candidates),5)):
        rows=candidates[start:start+5]
        rows.insert(page_no%(len(rows)+1),gold[page_no%len(gold)])
        pages.append(suite(main_id,rows,2,authors))
    upload_pages('review_pages',main_id,pages)
    by_count = {count: [t for t in candidates if len(json.loads(t['input_values']['shown_boxes'])) == count]
                for count in (0,1)}
    multiple = [t for t in candidates if len(json.loads(t['input_values']['shown_boxes'])) >= 2]
    assert len(multiple) >= 3
    pilot_pages = [suite(pilot_id, by_count[0][2*i:2*i+2] + by_count[1][2*i:2*i+2] + [multiple[i]], 2, authors)
                   for i in range(3)]
    upload_pages('pilot_pages',pilot_id,pilot_pages)
    upload_pages('missing_bbox_pages',bbox_id,[suite(bbox_id,[{'input_values':{'photo_id':int(missing['photo_id']),
                      'image_url':missing['image_url']}}],3)])
    assert len(list_items('tasks',pool_id=training_id))==6
    pools={name:request('GET',('trainings/' if name=='training' else 'pools/')+state[name])
           for name in ('training','exam','pilot','main','missing_bbox')}
    assert all(p['status']=='CLOSED' for p in pools.values())
    assert request('GET','requester')['balance']==account['balance']
    verification_path=HERE/'verification.json'
    previous=json.loads(verification_path.read_text()) if verification_path.exists() else {}
    save(verification_path,{**previous,'estimate_rub':estimate,'balance_rub':account['balance'],
          'excluded_first_stage_annotators':len(authors),
          'status':{name:p['status'] for name,p in pools.items()},'resources':state})
    print('Verified: all pools CLOSED, balance unchanged.',flush=True)


if __name__=='__main__':
    main()
