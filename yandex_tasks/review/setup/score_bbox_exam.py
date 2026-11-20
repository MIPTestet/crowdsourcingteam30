"""Validate Bbox exam submissions and grant the existing Bbox skill."""
import argparse
import csv
import json

from prepare import ROOT, list_items, request

SKILL = '21502'
EXAM_TSV = ROOT / 'yandex_tasks' / 'bbox' / 'tasks_exam.tsv'


def iou(a, b):
    left = max(a['left'], b['left'])
    top = max(a['top'], b['top'])
    right = min(a['left'] + a['width'], b['left'] + b['width'])
    bottom = min(a['top'] + a['height'], b['top'] + b['height'])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = a['width'] * a['height'] + b['width'] * b['height'] - intersection
    return intersection / union if union else 0


def matches(expected, submitted):
    matched = {}

    def augment(i, visited):
        for j, box in enumerate(submitted):
            if j in visited or iou(expected[i], box) < 0.5:
                continue
            visited.add(j)
            if j not in matched or augment(matched[j], visited):
                matched[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(expected)))


def score(assignment, truth):
    ids = [int(task['input_values']['photo_id']) for task in assignment['tasks']]
    solutions = assignment['solutions']
    if len(ids) != len(solutions) or len(ids) != len(truth) or set(ids) != set(truth):
        raise ValueError('Exam submission does not match the reference photos')
    correct = tp = total = 0
    negatives_correct = True
    no_load_errors = True
    for photo, solution in zip(ids, solutions, strict=True):
        expected_status, expected = truth[photo]
        answer = solution['output_values']
        submitted = answer.get('boxes') or []
        matched = matches(expected, submitted)
        correct += answer['status'] == expected_status and matched == len(expected) == len(submitted)
        tp += matched
        total += len(expected) + len(submitted)
        if expected_status == 'no_butterflies':
            negatives_correct &= answer['status'] == expected_status and not submitted
        no_load_errors &= answer['status'] != 'load_error'
    f1 = 2 * tp / total
    return correct >= 10 and f1 >= 0.85 and negatives_correct and no_load_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pool_id')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    pool = request('GET', 'pools/' + args.pool_id)
    if pool['project_id'] != '11838':
        raise ValueError('Expected the team Bbox project')
    with EXAM_TSV.open() as file:
        truth = {
            int(row['INPUT:photo_id']): (row['GOLDEN:status'], json.loads(row['GOLDEN:boxes']))
            for row in csv.DictReader(file, delimiter='\t')
        }
    results = list_items('assignments', pool_id=args.pool_id, status='ACCEPTED')
    skills = {s['user_id']: s['value'] for s in list_items('user-skills', skill_id=SKILL)}
    passed = {a['user_id'] for a in results if score(a, truth)}
    existing = {a['user_id'] for a in results if skills.get(a['user_id'], 0) >= 80}
    assessment_path = ROOT / 'results' / 'review' / 'bbox_extra_exam_assessment.json'
    assessment = json.loads(assessment_path.read_text()) if assessment_path.exists() else {}
    approved_assignments = set()
    if args.pool_id == assessment.get('pool_id'):
        approved_assignments.update(assessment['manually_approved_assignment_ids'])
    elif args.apply:
        raise ValueError('No manual assessment for this exam')
    approved_users = {a['user_id'] for a in results if a['id'] in approved_assignments}
    needs_grant = (passed & approved_users) - existing
    if args.apply:
        for worker in sorted(needs_grant):
            request('PUT', 'user-skills', {'skill_id': SKILL, 'user_id': worker, 'value': 100})
    print(json.dumps({'submissions': len(results), 'passed': len(passed),
                      'existing': len(existing), 'would_grant': len(needs_grant),
                      'passed_without_skill': len(passed - existing)}))


if __name__ == '__main__':
    main()
