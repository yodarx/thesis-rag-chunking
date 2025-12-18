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
        failures = 0
        max_failures = num_samples * 10  # Allow 10 failures per requested sample on average

        while len(dataset) < num_samples:
            sample = self.generate_sample(num_hops=num_hops)
            if sample:
                dataset.append(sample)
            else:
                failures += 1
                if failures >= max_failures:
                    print(
                        f"Warning: High failure rate. Stopped after {failures} failures. Generated {len(dataset)}/{num_samples} samples."
                    )
                    break

        return dataset

    def generate_sample(self, num_hops: int = 2) -> dict[str, Any] | None:
        chunks = self._get_random_contexts(n=num_hops)
        prompt = self._build_multihop_prompt(chunks)
        response = self.llm.invoke(prompt)
        qa_data = self._parse_llm_response(response)

        question = qa_data.get("question", "")
        answer = qa_data.get("answer", "")

        # Check for failure conditions
        if (
            question == "IMPOSSIBLE"
            or answer == "IMPOSSIBLE"
            or question == "Error parsing generation"
        ):
            return None

        return {
            "sample_id": str(uuid.uuid4()),
            "question": question,
            "answer": answer,
            "gold_passages": chunks,
            "category": "Multihop",
            "difficulty": "Hard",
        }

    def _get_random_contexts(self, n: int = 2, min_char_length: int = 100) -> list[str]:
        if not self.retriever.chunks:
            raise ValueError("Retriever has no chunks loaded.")

        # Filter out chunks that are too short (likely citations, headers, or noise)
        valid_chunks = [chunk for chunk in self.retriever.chunks if len(chunk) >= min_char_length]

        # Fallback: If filtering leaves us with too few chunks, revert to the full list
        if len(valid_chunks) < n:
            valid_chunks = self.retriever.chunks

        # Ensure we don't sample more than available
        n = min(n, len(valid_chunks))
        return random.sample(valid_chunks, n)

    def _build_multihop_prompt(self, chunks: list[str]) -> str:
        context_text = "\n\n".join([f"Context {i + 1}: {chunk}" for i, chunk in enumerate(chunks)])
        return (
            f"You are an expert at creating dataset questions and answers. "
            f"Given the following contexts, generate a single multi-hop question that requires information from ALL contexts to answer. "
            f"Also provide the correct answer based ONLY on the provided contexts. "
            f"The answer must be an EXACT substring found directly in one of the contexts (1:1 string match).\n"
            f"If the contexts are unrelated or it is not possible to form a valid multi-hop question, return 'IMPOSSIBLE' for both the question and answer.\n\n"
            f"CONTEXTS:\n{context_text}\n\n"
            f"Respond ONLY with a valid JSON object containing the keys 'question' and 'answer'. "
            f"Do not include any other text or markdown formatting."
        )

    def _parse_llm_response(self, response: str) -> dict[str, str]:
        cleaned_response = re.sub(r"```json\s*|\s*```", "", response).strip()
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            return {"question": "Error parsing generation", "answer": "Error parsing generation"}
