import json
from typing import Any

from langchain_community.llms import Ollama
from tqdm import tqdm


class DatasetCategorizer:
    def __init__(self, llm: Ollama | None = None):
        self.llm = llm

    def categorize_dataset(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        updated_dataset = []
        for data_point in tqdm(dataset, desc="Categorizing"):
            new_point = data_point.copy()
            if self.llm:
                category, difficulty = self._categorize_with_llm(new_point["question"])
                new_point["category"] = category
                new_point["difficulty"] = difficulty

            updated_dataset.append(new_point)
        return updated_dataset

    def _categorize_with_llm(self, question: str) -> tuple[str, str]:
        prompt = self._build_categorization_prompt(question)
        response = self.llm.invoke(prompt)
        return self._parse_llm_response(response)

    def _build_categorization_prompt(self, question: str) -> str:
        return (
            f"Categorize the following question into one of these categories and difficulties:\n"
            f"Categories: Factoid, Inference, Multihop\n"
            f"Difficulties: Easy, Moderate, Hard\n\n"
            f"Question: {question}\n\n"
            f"Respond in the format: Category | Difficulty\n"
            f"Example: Factoid | Easy"
        )

    def _parse_llm_response(self, response: str) -> tuple[str, str]:
        try:
            parts = response.strip().split("|")
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
        except Exception:
            pass
        return "Unknown", "Unknown"


def load_asqa_dataset(filepath: str | None, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        with open(filepath, encoding="utf-8") as f:
            all_data: list[dict[str, Any]] = [json.loads(line) for line in f]

        return all_data[:limit] if limit else all_data
    except FileNotFoundError:
        return []
