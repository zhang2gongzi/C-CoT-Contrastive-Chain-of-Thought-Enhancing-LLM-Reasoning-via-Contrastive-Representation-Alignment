import jsonlines
import re
import random
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from typing import List, Dict, Tuple

# --------------------------
# 配置参数（根据实际路径修改）
# --------------------------
DATASET_PATH = "/home2/zzl/ChatLogic/PARARULE-Plus/Depth3/PARARULE_Plus_Depth3_shuffled_dev_huggingface.jsonl"
MODEL_PATH = "/home2/zzl/model/Qwen2.5-7B-Instruct"
NUM_DEMOS = 4  # 4-shot演示（论文标准）
MAX_TEST_SAMPLES = 99  # 先小批量测试
OUTPUT_FILE = "/home2/zzl/C-CoT/baseline/ccotPrompting/depth3contrastive_cot_fixed_results.jsonl"
# Qwen模型推荐配置
MAX_NEW_TOKENS = 300  # 足够容纳多步骤推理
TEMPERATURE = 0.1     # 低随机性+非零，平衡严谨性与生成多样性
TOP_P = 0.9           # 辅助控制生成多样性
MAX_RECURSION_DEPTH = 5  # 最大递归重试次数，防止无限递归


# --------------------------
# 1. 自定义停止标准（修复过早截断问题）
# --------------------------
class StopStringsCriteria(StoppingCriteria):
    def __init__(self, stop_strings: List[str], tokenizer, min_gen_length: int = 50):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.min_gen_length = min_gen_length  # 最小生成长度，避免过早停止
        self.patterns = [re.compile(re.escape(s)) for s in stop_strings]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 生成长度不足时，不触发停止
        if input_ids.shape[1] < self.min_gen_length:
            return False
        # 生成长度足够时，检查停止字符串
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        for pattern in self.patterns:
            if pattern.search(generated_text):
                return True
        return False


# --------------------------
# 2. 数据处理工具函数（增强规则提取准确性）
# --------------------------
def split_context_into_facts_rules(context: str) -> Tuple[List[str], List[str]]:
    """拆分context为事实陈述(facts)和推理规则(rules)，适配数据集表述"""
    sentences = [s.strip() for s in context.split(". ") if s.strip()]
    facts = []
    rules = []
    # 更精准的规则关键词（覆盖数据集所有规则格式）
    rule_keywords = [
        "all ", "if ", 
        "strong people are", "thin and little then",
        "poor and rough then", "smart and quiet then",
        "short people are", "smart people are",
        "nice people are", "bad people are"
    ]

    for sent in sentences:
        lower_sent = sent.lower()
        # 规则：包含条件/全称量词
        if any(kw in lower_sent for kw in rule_keywords):
            rules.append(sent)
        # 事实：XX is XX（且不含规则关键词）
        elif " is " in lower_sent and not any(kw in lower_sent for kw in rule_keywords):
            facts.append(sent)
    return facts, rules


# --------------------------
# 3. 有效推理演示(T₊)生成（增加递归限制）
# --------------------------
def generate_valid_rationale(
    context: str, 
    question: str, 
    label: int, 
    tokenizer, 
    model,
    recursion_depth: int = 0  # 新增：递归深度计数器
) -> str:
    """生成高质量有效推理：明确要求3+步骤，使用Qwen推荐指令格式"""
    # 新增：递归深度检查，超过限制则终止
    if recursion_depth >= MAX_RECURSION_DEPTH:
        print(f"Warning: Reached maximum recursion depth for question '{question}'. Using fallback rationale.")
        # 返回保底推理（手动构造符合格式的推理）
        correct_answer = "Yes" if label == 1 else "No"
        return f"Step 1: Analyze the given context. Step 2: Apply relevant rules. Step 3: Thus, the statement '{question}' is {correct_answer.lower()}."

    facts, rules = split_context_into_facts_rules(context)
    facts_str = "\n  - " + "\n  - ".join(facts) if facts else "  - No facts"
    rules_str = "\n  - " + "\n  - ".join(rules) if rules else "  - No rules"
    correct_answer = "Yes" if label == 1 else "No"

    # Qwen指令格式：系统提示+用户输入，明确要求3+推理步骤
    prompt = f"""<|system|>
You are a logical reasoning expert. For the given facts, rules, and question, you must:
1. Write a step-by-step explanation with AT LEAST 3 steps (each step starts with "Step X: "). 
2. Each step must clearly connect facts to rules or intermediate conclusions. 
3. The final step must state whether the question is true or false.
<|user|>
Given:
- Facts:{facts_str}
- Rules:{rules_str}
Question: Is the statement "{question}" true?
Correct Answer: {correct_answer}
Please provide the step-by-step explanation:
Explanation: <|endoftext|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # 停止标准：最小生成长度50，避免短文本截断
    stopping_criteria = StoppingCriteriaList([ 
        StopStringsCriteria(stop_strings=["<|endoftext|>", "Question:"], 
                            tokenizer=tokenizer, min_gen_length=50)
    ])

    # 调整生成参数，提高多样性
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=GenerationConfig(
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,  # 开启采样以提高多样性
                temperature=0.7,  # 增加温度，生成更多样化的推理
                top_p=0.95,  # 控制生成的多样性
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            ),
            stopping_criteria=stopping_criteria
        )

    # 解析并清理推理结果
    rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取<|user|>后的Explanation部分
    rationale = rationale.split("Explanation: ")[-1].strip()
    # 移除可能的停止符
    for stop in ["<|endoftext|>", "Question:", "Answer:"]:
        rationale = rationale.replace(stop, "").strip()

    # 验证推理质量：至少3个步骤
    if len(re.findall(r"Step \d+:", rationale)) < 3:
        # 递归重试，增加深度计数
        return generate_valid_rationale(context, question, label, tokenizer, model, recursion_depth + 1)
    return rationale


def generate_valid_demos(
    dataset_path: str, 
    tokenizer, 
    model, 
    num_demos: int = 4
) -> List[str]:
    """生成有效演示并打印验证（确保演示逻辑正确）"""
    valid_demos = []
    print("=== Generating Valid Demonstrations (T₊) ===")
    with jsonlines.open(dataset_path, "r") as f:
        for line in f:
            if len(valid_demos) >= num_demos:
                break
            context = line["context"]
            question = line["question"]
            label = line["label"]
            correct_answer = "Yes" if label == 1 else "No"

            # 生成有效推理（不传递递归深度，使用默认值0）
            valid_rationale = generate_valid_rationale(context, question, label, tokenizer, model)
            # 格式化演示（清晰易读）
            demo = f"""Example {len(valid_demos)+1}:
Given:
  Context: {context}
  Question: Is "{question}" true?
Explanation (Correct):
  {valid_rationale}
Answer (Correct): {correct_answer}
"""
            # 打印验证：确保演示可肉眼判断逻辑正确
            print(f"Generated Valid Demo {len(valid_demos)+1}:\n{demo}")
            valid_demos.append(demo)
    return valid_demos


# --------------------------
# 4. 无效推理演示(T₋)生成（精准破坏逻辑链）
# --------------------------
def extract_logic_bridges(context: str, valid_rationale: str) -> List[str]:
    """提取推理中的关键逻辑桥接（步骤-事实-规则关联）"""
    facts, rules = split_context_into_facts_rules(context)
    logic_bridges = []

    # 提取步骤中的事实关联（如"Step 1: Use fact 'Harry is strong'"）
    fact_mentions = re.findall(r"Step \d+: .*?'(.+? is .+?)'", valid_rationale)
    for fact in fact_mentions:
        if fact in facts:
            logic_bridges.append(f"Fact: '{fact}'")

    # 提取步骤中的规则关联（如"Step 2: Apply rule 'Strong people are smart'"）
    rule_mentions = re.findall(r"Step \d+: .*?'(.+?are .+?)'", valid_rationale)
    for rule in rule_mentions:
        for full_rule in rules:
            if rule in full_rule:
                logic_bridges.append(f"Rule: '{full_rule}'")

    # 提取步骤中的中间结论（如"Step 3: Thus, Harry is smart"）
    intermediate_conclusions = re.findall(r"Step \d+: .*?Thus, (.+? is .+?)(\.|$)", valid_rationale)
    logic_bridges.extend([f"Conclusion: '{conc.strip()}'" for conc, _ in intermediate_conclusions])

    # 保底：确保至少2个桥接对象
    if len(logic_bridges) < 2:
        step_fragments = re.findall(r"Step \d+: (.+?)(Step \d+:|$)", valid_rationale, re.DOTALL)
        logic_bridges.extend([f"Step: '{frag.strip()}'" for frag, _ in step_fragments if frag.strip()])

    return list(set(logic_bridges))[:6]  # 限制最大6个，避免过度混乱


def generate_invalid_rationale(
    context: str, 
    question: str, 
    valid_rationale: str, 
    label: int
) -> str:
    """生成无效推理：破坏步骤顺序+错误关联事实规则，保留步骤格式"""
    logic_bridges = extract_logic_bridges(context, valid_rationale)
    correct_answer = "Yes" if label == 1 else "No"
    wrong_answer = "No" if label == 1 else "Yes"

    # 1. 打乱逻辑桥接对象
    shuffled_bridges = logic_bridges.copy()
    random.shuffle(shuffled_bridges)

    # 2. 替换有效推理中的桥接对象，生成逻辑错误
    invalid_rationale = valid_rationale
    for orig, shuf in zip(logic_bridges, shuffled_bridges):
        if orig in invalid_rationale:
            invalid_rationale = invalid_rationale.replace(orig, shuf)

    # 3. 修改最终结论为错误答案（修复f-string反斜杠问题）
    step_pattern = r"Step \d+: .*?the statement .*? is (true|false)"
    step_count = len(re.findall(r'Step \d+:', invalid_rationale))
    replacement = f"Step {step_count}: Thus, the statement '{question}' is {wrong_answer.lower()}"
    invalid_rationale = re.sub(step_pattern, replacement, invalid_rationale)

    # 打印验证：确保无效推理逻辑错误但格式正确
    print(f"Generated Invalid Rationale:\n  {invalid_rationale}\n")
    return invalid_rationale


# --------------------------
# 5. 模型推理（Qwen专属指令格式+结果解析优化）
# --------------------------
def build_contrastive_prompt(contrastive_demos: List[str], test_sample: Dict) -> str:
    """构建对比CoT提示：使用Qwen格式，明确学习对比逻辑"""
    system_prompt = """<|system|>
You are a logical reasoning expert. Learn from the following examples (each has a CORRECT and a WRONG explanation):
1. First, understand why the CORRECT explanation is logical (follows facts → rules → conclusions).
2. Then, avoid the mistakes in the WRONG explanation (e.g., shuffled steps, wrong fact-rule connections).
3. When solving the new problem, write a step-by-step explanation with AT LEAST 3 steps (start with "Step X: ").
4. You MUST end your explanation EXACTLY with: "Thus, the answer is Yes." or "Thus, the answer is No."
<|user|>
"""
    demos_str = "\n".join(contrastive_demos)
    test_context = test_sample["context"]
    test_question = test_sample["question"]

    test_prompt = f"""
{demos_str}

Now solve the NEW problem:
Given:
  Context: {test_context}
Question: Is "{test_question}" true?
Please provide your reasoning in this format:

Step 1: ...
Step 2: ...
Step 3: ...
Thus, the answer is Yes/No.
<|endoftext|>"""

    return system_prompt + test_prompt


def parse_generated_result(generated_text: str, test_question: str) -> Tuple[str, str]:
    """优化结果解析：精准提取推理和答案"""
    # 提取 Explanation 部分
    expl_start = generated_text.find("Explanation: ")
    if expl_start != -1:
        expl_start += len("Explanation: ")
    else:
        expl_start = 0
    expl_end = generated_text.find("<|endoftext|>", expl_start)
    generated_expl = generated_text[expl_start:expl_end].strip() if expl_end != -1 else generated_text[expl_start:].strip()

    # 提取答案
    answer_match = re.search(r"Thus, the answer is (Yes|No)", generated_expl, re.IGNORECASE)
    if answer_match:
        generated_ans = answer_match.group(1).capitalize()
    else:
        # 兜底逻辑
        if " is true" in generated_expl.lower():
            generated_ans = "Yes"
        elif " is false" in generated_expl.lower():
            generated_ans = "No"
        else:
            generated_ans = "Unknown"

    # 清理多余字符
    for char in ["<|endoftext|>", "Question:", "Answer:"]:
        generated_expl = generated_expl.replace(char, "").strip()

    return generated_expl, generated_ans


def run_contrastive_cot_inference():
    """运行对比CoT推理并保存结果"""
    # 加载模型和分词器
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # 确保分词器有pad_token（部分模型默认没有）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    ).eval()  # 推理模式

    # 生成有效演示和对比演示
    print("Generating valid demonstrations...")
    valid_demos = generate_valid_demos(DATASET_PATH, tokenizer, model, NUM_DEMOS)

    print("Building contrastive demonstrations...")
    contrastive_demos = build_contrastive_demos(
        valid_demos, DATASET_PATH, tokenizer, model, NUM_DEMOS
    )

    # 读取测试样本（跳过演示样本）
    print("Loading test samples...")
    test_samples = []
    with jsonlines.open(DATASET_PATH, "r") as f:
        all_samples = [line for line in f]
        test_samples = all_samples[NUM_DEMOS:NUM_DEMOS+MAX_TEST_SAMPLES]

    # 批量推理
    print(f"Starting inference on {len(test_samples)} samples...")
    results = []
    for idx, test_sample in enumerate(test_samples):
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(test_samples)} samples")

        # 构建提示词
        prompt = build_contrastive_prompt(contrastive_demos, test_sample)

        # 模型生成
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 定义停止字符串并创建停止标准
        stopping_criteria = StoppingCriteriaList([
            StopStringsCriteria(stop_strings=["<|endoftext|>", "Example:", "Given:"], 
                                tokenizer=tokenizer, min_gen_length=80)  # 推理至少80个token
        ])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.pad_token_id
                ),
                stopping_criteria=stopping_criteria
            )

        # 解析结果
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_expl, generated_ans = parse_generated_result(generated_text, test_sample["question"])

        # 保存结果
        results.append({
            "id": test_sample["id"],
            "question": test_sample["question"],
            "true_label": test_sample["label"],
            "true_answer": "Yes" if test_sample["label"] == 1 else "No",
            "generated_explanation": generated_expl,
            "generated_answer": generated_ans,
            "is_correct": (generated_ans == "Yes" and test_sample["label"] == 1) or 
                          (generated_ans == "No" and test_sample["label"] == 0)
        })

    # 保存结果到文件
    with jsonlines.open(OUTPUT_FILE, "w") as f:
        f.write_all(results)

    # 计算准确率
    accuracy = sum(1 for res in results if res["is_correct"]) / len(results)
    print(f"Inference completed! Accuracy: {accuracy:.4f}")
    print(f"Results saved to {OUTPUT_FILE}")

def build_contrastive_demos(
    valid_demos: List[str],
    dataset_path: str,
    tokenizer,
    model,
    num_demos: int = 4
) -> List[str]:
    """构建对比演示 (T₊ vs T₋)，即每个正确推理 + 对应错误推理"""
    contrastive_demos = []
    with jsonlines.open(dataset_path, "r") as f:
        all_samples = [line for line in f]

    for i, demo in enumerate(valid_demos):
        sample = all_samples[i]
        context = sample["context"]
        question = sample["question"]
        label = sample["label"]
        correct_answer = "Yes" if label == 1 else "No"

        # 从 valid_demo 里提取正确推理
        valid_rationale_match = re.search(r"Explanation \(Correct\):\s*(.+?)\nAnswer", demo, re.DOTALL)
        if not valid_rationale_match:
            print(f"Warning: Failed to extract rationale from demo {i+1}")
            continue
        valid_rationale = valid_rationale_match.group(1).strip()

        # 生成错误推理
        invalid_rationale = generate_invalid_rationale(context, question, valid_rationale, label)

        # 拼接对比演示
        contrastive_demo = f"""Example {i+1}:
Given:
  Context: {context}
  Question: Is "{question}" true?
Explanation (CORRECT):
  {valid_rationale}
Answer (Correct): {correct_answer}

Explanation (WRONG):
  {invalid_rationale}
Answer (Wrong): {"No" if correct_answer=="Yes" else "Yes"}
"""
        contrastive_demos.append(contrastive_demo)

        if len(contrastive_demos) >= num_demos:
            break

    return contrastive_demos


# --------------------------
# 6. 主函数入口
# --------------------------
if __name__ == "__main__":
    # 新增：增加Python递归深度限制（默认递归深度较浅）
    import sys
    sys.setrecursionlimit(10000)  # 设置为10000，足够应对正常场景
    run_contrastive_cot_inference()