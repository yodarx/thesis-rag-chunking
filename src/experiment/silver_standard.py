import json
import random
import re
import uuid
from typing import Any

from langchain_community.llms import Ollama

from src.experiment.retriever import FaissRetriever


class SilverStandardGenerator:
    def __init__(self, retriever: FaissRetriever, llm: Ollama):
        self.retriever = retriever
        self.llm = llm

    def generate_dataset(self, num_samples: int, num_hops: int = 2) -> list[dict[str, Any]]:
        dataset = []
        for _ in range(num_samples):
            dataset.append(self.generate_sample(num_hops=num_hops))
        return dataset

    def generate_sample(self, num_hops: int = 2) -> dict[str, Any]:
        chunks = self._get_random_contexts(n=num_hops)
        prompt = self._build_multihop_prompt(chunks)
        response = self.llm.invoke(prompt)
        qa_data = self._parse_llm_response(response)

        return {
            "sample_id": str(uuid.uuid4()),
            "question": qa_data.get("question", ""),
            "answer": qa_data.get("answer", ""),
            "gold_passages": chunks,
            "category": "Multihop",
            "difficulty": "Hard",
        }

    def _get_random_contexts(self, n: int = 2) -> list[str]:
        if not self.retriever.chunks:
            raise ValueError("Retriever has no chunks loaded.")
        # Ensure we don't sample more than available
        n = min(n, len(self.retriever.chunks))
        return random.sample(self.retriever.chunks, n)

    def _build_multihop_prompt(self, chunks: list[str]) -> str:
        context_text = "\n\n".join([f"Context {i + 1}: {chunk}" for i, chunk in enumerate(chunks)])
        return (
            f"You are an expert at creating dataset questions and answers. "
            f"Given the following contexts, generate a single multi-hop question that requires information from ALL contexts to answer. "
            f"Also provide the correct answer based ONLY on the provided contexts The correct answert needs to be in 1:1 in the context. CONTEXT:.\n\n"
            f"{context_text}\n\n"
            f"Respond ONLY with a valid JSON object containing the keys 'question' and 'answer'. "
            f"Do not include any other text or markdown formatting."
        )

    def _parse_llm_response(self, response: str) -> dict[str, str]:
        cleaned_response = re.sub(r"```json\s*|\s*```", "", response).strip()
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            return {"question": "Error parsing generation", "answer": "Error parsing generation"}
