import json, re, os, sys

def convert_gsm8k(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = []
    for i, item in enumerate(data):
        q = item['question']
        gt = item.get('ground_truth', '')
        m = re.search(r'####\s*(-?[\d,\.]+)', gt)
        gold = m.group(1).replace(',', '') if m else None
        if gold is None:
            continue
        qid = f'gsm8k_llama_{i}'
        for path in item.get('paths', []):
            cot = path['reasoning']
            pred = str(path.get('predicted_answer', '')).strip()
            try:
                is_correct = int(abs(float(pred) - float(gold)) < 1e-6)
            except:
                is_correct = int(pred.strip().lower() == gold.strip().lower())
            out.append(json.dumps({
                'raw_example': {'id': qid, 'question': q, 'label': gold},
                'cot': cot, 'gold_label': gold, 'is_correct': is_correct
            }, ensure_ascii=False))
    with open(output_path, 'w', encoding='utf-8') as f:
        for l in out:
            f.write(l + '\n')
    uids = set(json.loads(l)['raw_example']['id'] for l in out)
    print(f'  output: {len(out)} paths, {len(uids)} questions')


def convert_csqa(input_path, qwen_ref_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    qwen_map = {}
    with open(qwen_ref_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            q = obj['raw_example']['question']
            if q not in qwen_map:
                qwen_map[q] = obj['gold_label']
    print(f'  qwen ref: {len(qwen_map)} unique questions')

    out = []
    skipped = 0
    for i, item in enumerate(data):
        q = item['question']
        gold = qwen_map.get(q)
        if gold is None:
            skipped += 1
            continue
        qid = f'csqa_llama_{i}'
        for path in item.get('paths', []):
            cot = path['reasoning']
            pred = str(path.get('predicted_answer', '')).strip()
            is_correct = 1 if (pred == gold or gold in pred) else 0
            out.append(json.dumps({
                'raw_example': {'id': qid, 'question': q, 'label': gold},
                'cot': cot, 'gold_label': gold, 'is_correct': is_correct
            }, ensure_ascii=False))
    if skipped:
        print(f'  skipped {skipped} questions (no match)')
    with open(output_path, 'w', encoding='utf-8') as f:
        for l in out:
            f.write(l + '\n')
    uids = set(json.loads(l)['raw_example']['id'] for l in out)
    print(f'  output: {len(out)} paths, {len(uids)} questions')


def convert_strategyqa(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = []
    for i, item in enumerate(data):
        q = item['question']
        gold = item.get('ground_truth', '').strip().lower()
        if gold not in ('yes', 'no'):
            continue
        qid = f'strategyqa_llama_{i}'
        for path in item.get('paths', []):
            cot = path['reasoning']
            pred_raw = str(path.get('predicted_answer', '')).strip().lower()
            # Extract yes/no from prediction
            has_yes = bool(re.search(r'\byes\b', pred_raw))
            has_no = bool(re.search(r'\bno\b', pred_raw))
            if has_yes and not has_no:
                pred = 'yes'
            elif has_no and not has_yes:
                pred = 'no'
            elif pred_raw.startswith('yes'):
                pred = 'yes'
            elif pred_raw.startswith('no'):
                pred = 'no'
            else:
                pred = pred_raw
            is_correct = int(pred == gold)
            out.append(json.dumps({
                'raw_example': {'id': qid, 'question': q, 'label': gold},
                'cot': cot, 'gold_label': gold, 'is_correct': is_correct
            }, ensure_ascii=False))
    with open(output_path, 'w', encoding='utf-8') as f:
        for l in out:
            f.write(l + '\n')
    uids = set(json.loads(l)['raw_example']['id'] for l in out)
    print(f'  output: {len(out)} paths, {len(uids)} questions')


BASE = 'E:/dporot/C-CoT-Contrastive-Chain-of-Thought-Enhancing-LLM-Reasoning-via-Contrastive-Representation-Alignment'
DATADIR = os.path.join(BASE, 'baseline/ORM/data')
OUTDIR = os.path.join(BASE, 'newrundata')

print('=== GSM8K LLaMA ===')
convert_gsm8k(
    os.path.join(DATADIR, 'gsm8k_llama_1k/ckpt_generated.json'),
    os.path.join(OUTDIR, 'gsm8k_llama_flat.jsonl')
)

print('=== CSQA LLaMA ===')
convert_csqa(
    os.path.join(DATADIR, 'csqa_llama_420/all_generated.json'),
    os.path.join(OUTDIR, 'csqa_500_flat.jsonl'),
    os.path.join(OUTDIR, 'csqa_llama_flat.jsonl')
)

print('=== StrategyQA LLaMA ===')
convert_strategyqa(
    os.path.join(DATADIR, 'strategyqa_llama_420/all_generated.json'),
    os.path.join(OUTDIR, 'strategyqa_llama_flat.jsonl')
)

print('\nDone! Three LLaMA flat jsonl files saved to newrundata/')
