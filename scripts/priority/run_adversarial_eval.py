"""
run_adversarial_eval.py
────────────────────────
Runs the 100 hand-authored synthetic emails (adversarial_test_set.py) through
the live EmailClassifier and scores predicted category/priority against the
ground truth assigned at authoring time. Sentiment (VADER compound) is
reported diagnostically against expected_sentiment but NOT scored as
pass/fail, since it isn't a final classify() output.

Usage
  python run_adversarial_eval.py --models ../../models
"""

import argparse
import sys

sys.path.insert(0, (__file__.rsplit('\\', 1)[0] or '.') + '\\..')
sys.path.insert(0, (__file__.rsplit('\\', 1)[0] or '.') + '\\..\\shared')

from classify_email import EmailClassifier, _sia, _clean
from adversarial_test_set import TEST_EMAILS


def sentiment_bucket(compound):
    if compound >= 0.05:
        return 'positive'
    if compound <= -0.05:
        return 'negative'
    return 'neutral'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='../../models')
    args = ap.parse_args()

    clf = EmailClassifier(args.models)

    rows = []
    for e in TEST_EMAILS:
        result = clf.classify(subject=e['subject'], body=e['body'], to=e['to'], from_addr=e['from_addr'])
        text = _clean(e['subject'] + ' ' + e['body'])
        compound = _sia.polarity_scores(text)['compound']
        rows.append({
            'id': e['id'],
            'note': e['note'],
            'exp_cat': e['expected_category'], 'pred_cat': result['category']['label'],
            'exp_prio': e['expected_priority'], 'pred_prio': result['priority']['bucket'],
            'pred_score': result['priority']['score'],
            'exp_sent': e['expected_sentiment'], 'pred_sent': sentiment_bucket(compound),
            'compound': compound,
            'spam_flag': result['spam']['label'],
        })

    n = len(rows)
    cat_correct = sum(1 for r in rows if r['exp_cat'] == r['pred_cat'])
    prio_correct = sum(1 for r in rows if r['exp_prio'] == r['pred_prio'])
    sent_correct = sum(1 for r in rows if r['exp_sent'] == r['pred_sent'])

    print(f"\n=== Adversarial test set: {n} hand-authored synthetic emails ===\n")
    print(f"Category accuracy:  {cat_correct}/{n} = {cat_correct/n:.1%}")
    print(f"Priority accuracy:  {prio_correct}/{n} = {prio_correct/n:.1%}")
    print(f"Sentiment agreement (informational, VADER sign vs hand label): {sent_correct}/{n} = {sent_correct/n:.1%}")

    print("\n--- Priority: per-expected-bucket breakdown ---")
    for bucket in ['low', 'medium', 'high']:
        sub = [r for r in rows if r['exp_prio'] == bucket]
        if not sub:
            continue
        c = sum(1 for r in sub if r['pred_prio'] == bucket)
        print(f"  {bucket:6s}: {c}/{len(sub)} = {c/len(sub):.1%}")

    print("\n--- Priority confusion matrix (rows=expected, cols=predicted) ---")
    buckets = ['low', 'medium', 'high']
    header = "        " + "  ".join(f"{b:>7s}" for b in buckets)
    print(header)
    for eb in buckets:
        counts = [sum(1 for r in rows if r['exp_prio'] == eb and r['pred_prio'] == pb) for pb in buckets]
        print(f"  {eb:6s}" + "".join(f"{c:9d}" for c in counts))

    print("\n--- Category: per-expected-class breakdown ---")
    for cls in ['Work', 'Personal', 'Other']:
        sub = [r for r in rows if r['exp_cat'] == cls]
        if not sub:
            continue
        c = sum(1 for r in sub if r['pred_cat'] == cls)
        print(f"  {cls:9s}: {c}/{len(sub)} = {c/len(sub):.1%}")

    print("\n--- Priority misses (expected != predicted) ---")
    for r in rows:
        if r['exp_prio'] != r['pred_prio']:
            print(f"  [{r['id']}] exp={r['exp_prio']:6s} pred={r['pred_prio']:6s} (score={r['pred_score']:.2f})  "
                  f"cat_ok={'Y' if r['exp_cat']==r['pred_cat'] else 'N'}  spam={r['spam_flag']}  -- {r['note']}")

    print("\n--- Category misses (expected != predicted) ---")
    for r in rows:
        if r['exp_cat'] != r['pred_cat']:
            print(f"  [{r['id']}] exp={r['exp_cat']:9s} pred={r['pred_cat']:9s}  -- {r['note']}")

    print("\n--- Sentiment mismatches (informational only) ---")
    for r in rows:
        if r['exp_sent'] != r['pred_sent']:
            print(f"  [{r['id']}] exp={r['exp_sent']:8s} pred={r['pred_sent']:8s} (compound={r['compound']:+.2f})  -- {r['note']}")


if __name__ == '__main__':
    main()
