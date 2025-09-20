# path_generator.py
from typing import List, Tuple
import re
from tqdm import tqdm

class PathGenerator:
    def __init__(self, config, llm_client):
        self.config = config
        self.llm = llm_client

    def generate_single_path(self, question: str) -> Tuple[str, str]:
        system_msg = "You are a helpful assistant. Think step by step."
        inst = f"{system_msg}\n\nQuestion: {question}\nThought:"
        prompt = f"<s>[INST] {inst} [/INST]"
        
        response = self.llm.generate(prompt, self.config.max_new_tokens, self.config.temperature)
        full_reasoning = f"Thought: {response}"
        answer = self.extract_answer(full_reasoning)
        return full_reasoning, answer

    def extract_answer(self, text: str) -> str:
        lines = text.strip().split('\n')
        for line in reversed(lines):
            if 'answer is' in line.lower():
                match = re.search(r'answer is\s+(yes|no)', line, re.IGNORECASE)
                if match:
                    return match.group(1).lower()
        return "unknown"

    def generate_multi_paths(self, dataset) -> List[dict]:
        multi_data = []
        for item in tqdm(dataset, desc="🧠 生成多路径中", total=len(dataset)):
            paths = []
            for _ in range(self.config.num_paths):
                reasoning, answer = self.generate_single_path(item["question"])
                is_positive = (answer == item["gold_answer"])
                paths.append({"reasoning": reasoning, "answer": answer, "is_positive": is_positive})
            multi_data.append({
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "paths": paths
            })
        return multi_data