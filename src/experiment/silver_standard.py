import json
import random
import re
import uuid
from typing import Any


class SilverStandardGenerator:
    """
    Generates multi-hop QA pairs by randomly selecting 3 Wikipedia articles
    and prompting an LLM (Gemini or Ollama) to create a question requiring
    information from all selected articles.
    """

    def __init__(self, llm_client: Any, documents: list[str]):
        """
        Initialize the silver standard generator.

        Args:
            llm_client: Either a Google genai.Client (for Gemini) or Ollama client
            documents: List of document texts to sample from"""
        self.llm_client = llm_client
        self.model = "gemini-3.0-preview"
        self.documents = documents

    def generate_dataset(self, num_samples: int, num_hops: int = 3) -> list[dict[str, Any]]:
        """Generate a dataset of multi-hop QA pairs."""
        dataset = []
        failures = 0
        max_failures = num_samples * 10

        while len(dataset) < num_samples:
            sample = self.generate_sample(num_hops=num_hops)
            if sample:
                dataset.append(sample)
            else:
                failures += 1
                if failures >= max_failures:
                    print(
                        f"Warning: High failure rate. Stopped after {failures} failures. "
                        f"Generated {len(dataset)}/{num_samples} samples."
                    )
                    break

        return dataset

    def generate_sample(self, num_hops: int = 3) -> dict[str, Any] | None:
        """Generate a single multi-hop QA pair with evidence snippets."""
        try:
            chunks = self._get_random_contexts(n=num_hops)

            # 2. Build and send prompt
            prompt = self._build_multihop_prompt(chunks)
            response_text = self._call_llm(prompt)
            qa_data = self._parse_llm_response(response_text)

            question = qa_data.get("question", "")
            answer = qa_data.get("answer", "")
            gold_snippets = qa_data.get("gold_snippets", [])

            if (
                    question == "IMPOSSIBLE"
                    or answer == "IMPOSSIBLE"
                    or question == "Error parsing generation"
                    or not gold_snippets
            ):
                return None

            return {
                "sample_id": str(uuid.uuid4()),
                "question": question,
                "answer": answer,
                "gold_passages": gold_snippets,
                "category": "Multihop",
                "difficulty": "Hard",
                "metadata": {
                    "source_chunks": chunks,
                    "bridge_entity": qa_data.get("bridge_entity"),
                },
            }
        except Exception as e:
            print(f"Error generating sample: {e}")
            return None

    def _get_random_contexts(self, n: int = 3, min_char_length: int = 100) -> list[str]:
        candidates = [doc for doc in self.documents if len(doc) >= min_char_length]
        return random.sample(candidates, n) if len(candidates) >= n else candidates

    def _build_multihop_prompt(self, contexts: list[str]) -> str:
        """Build the prompt for multi-hop question generation with strict bridge validation."""
        context_text = "\n\n".join([f"Context {i + 1}: {chunk}" for i, chunk in enumerate(contexts)])

        return (
            f"You are an expert at creating high-quality logical reasoning datasets. "
            f"Your task is to generate a 'multi-hop' question based on the provided contexts.\n\n"
            f"### DEFINITION OF A VALID MULTI-HOP QUESTION:\n"
            f"A valid multi-hop question requires the reader to traverse a 'Bridge Entity' to find the answer. "
            f"It implies a dependency: You need Fact A to understand the Question, and Fact B to find the Answer.\n"
            f"1. **Identify a Bridge:** Find a SPECIFIC entity (person, organization, unique event) that is explicitly mentioned in **Context A** and also appears in **Context B**.\n"
            f"2. **Formulate the Question:** Ask for a specific attribute of the entity found in Context B, but refer to that entity *only* by its description or role from Context A.\n"
            f"   - BAD (Single Hop): 'What is the cost of the ACT Proof of Identity Card?' (This only relies on one context. REJECT.)\n"
            f"   - GOOD (Multi Hop): 'In which city is the company that currently finishes the official FFA jackets located?' (Requires Context A to identify the company, Context B to find its location.)\n\n"
            f"### STRICT VALIDATION RULES (READ CAREFULLY):\n"
            f"1. **The 'Bridge Test':** If you were to delete Context A, the question MUST become unanswerable or ambiguous. If the question still makes sense without Context A, it is INVALID. Return 'IMPOSSIBLE'.\n"
            f"2. **Specific vs. Generic:** The Bridge Entity must be the SAME specific instance. \n"
            f"   - 'Birth Certificate' (US) and 'Birth Certificate' (Australia) are generic concepts, not the same entity. -> IMPOSSIBLE.\n"
            f"   - 'The 1938 edition of Understanding Poetry' (Context A) and 'Understanding Poetry' (Context B) are the same entity. -> VALID.\n"
            f"3. **No Hallucinations:** The answer must be 100% supported by the text. Do not use outside knowledge.\n\n"
            f"### CRITICAL REQUIREMENT: EVIDENCE EXTRACTION\n"
            f"You must extract the EXACT sentences (snippets) from the text that prove the logic works.\n"
            f"- **Snippet 1:** The exact sentence from Context A that links the description to the Bridge Entity.\n"
            f"- **Snippet 2:** The exact sentence from Context B that provides the answer about that Bridge Entity.\n\n"
            f"### INPUT CONTEXTS:\n"
            f"{context_text}\n\n"
            f"### OUTPUT FORMAT:\n"
            f"Respond ONLY with a valid JSON object. Do not add markdown blocks.\n"
            f"If a valid question cannot be formed (which is likely for random articles), set all values to 'IMPOSSIBLE' (and empty list for snippets).\n"
            f"{{\n"
            f'  "bridge_entity": "The specific entity connecting the two texts",\n'
            f'  "question": "The multi-hop question",\n'
            f'  "answer": "The exact substring answer",\n'
            f'  "gold_snippets": ["Exact sentence from Context A", "Exact sentence from Context B"]\n'
            f"}}"
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        return self._call_gemini(prompt)

    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API."""
        try:
            response = self.llm_client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise

    def _parse_llm_response(self, response: str) -> dict[str, str]:
        """Parse JSON response from LLM."""
        cleaned_response = re.sub(r"```json\s*|\s*```", "", response).strip()
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            return {"question": "Error parsing generation", "answer": "Error parsing generation"}
