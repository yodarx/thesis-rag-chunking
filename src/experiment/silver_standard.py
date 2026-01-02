import json
import random
import re
import uuid
from typing import Any

from src.experiment.retriever import FaissRetriever


class SilverStandardGenerator:
    """
    Generates multi-hop QA pairs by randomly selecting 3 Wikipedia articles
    and prompting an LLM (Gemini or Ollama) to create a question requiring
    information from all selected articles.
    """

    def __init__(self, retriever: FaissRetriever, llm_client: Any, llm_type: str = "gemini"):
        """
        Initialize the silver standard generator.

        Args:
            retriever: FaissRetriever instance with chunks loaded
            llm_client: Either a Google genai.Client (for Gemini) or Ollama client
            llm_type: "gemini" for Google Gemini API or "ollama" for local Ollama
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.llm_type = llm_type.lower()

        if self.llm_type == "gemini":
            self.model = "gemini-3.0-preview"
        elif self.llm_type == "ollama":
            self.model = None  # Ollama client handles model selection
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}. Use 'gemini' or 'ollama'.")

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
            # 1. Get random raw contexts (e.g., full paragraphs or pages)
            # Note: Ideally, these are NOT pre-chunked by your specific strategy,
            # but are larger raw sections to allow for unbiased snippet extraction.
            chunks = self._get_random_contexts(n=num_hops)

            # 2. Build and send prompt
            prompt = self._build_multihop_prompt(chunks)
            response_text = self._call_llm(prompt)
            qa_data = self._parse_llm_response(response_text)

            question = qa_data.get("question", "")
            answer = qa_data.get("answer", "")
            gold_snippets = qa_data.get("gold_snippets", [])

            # 3. Check for failure conditions
            if (
                question == "IMPOSSIBLE"
                or answer == "IMPOSSIBLE"
                or question == "Error parsing generation"
                or not gold_snippets  # Fail if no evidence was extracted
            ):
                return None

            # 4. Return the valid sample
            # We store 'gold_snippets' as 'gold_passages' because these are the
            # actual Ground Truth texts required for evaluation.
            return {
                "sample_id": str(uuid.uuid4()),
                "question": question,
                "answer": answer,
                "gold_passages": gold_snippets,  # The specific sentences, not the random blocks
                "category": "Multihop",
                "difficulty": "Hard",
                "metadata": {
                    # Optional: Store the original chunks if you want to debug later
                    "source_chunks": chunks,
                    "bridge_entity": qa_data.get("bridge_entity"),
                },
            }
        except Exception as e:
            print(f"Error generating sample: {e}")
            return None

    def _get_random_contexts(self, n: int = 3, min_char_length: int = 100) -> list[str]:
        """Randomly select n chunks from the dataset."""
        if not self.retriever.chunks:
            raise ValueError("Retriever has no chunks loaded.")

        # Simple random selection without semantic similarity
        available_chunks = [
            chunk for chunk in self.retriever.chunks if len(chunk) >= min_char_length
        ]

        if not available_chunks:
            # Fallback: use all chunks if none meet length requirement
            available_chunks = self.retriever.chunks

        # Randomly select n chunks
        selected_chunks = random.sample(available_chunks, min(n, len(available_chunks)))

        return selected_chunks

    def _build_multihop_prompt(self, chunks: list[str]) -> str:
        """Build the prompt for multi-hop question generation with strict bridge validation."""
        context_text = "\n\n".join([f"Context {i + 1}: {chunk}" for i, chunk in enumerate(chunks)])

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
        if self.llm_type == "gemini":
            return self._call_gemini(prompt)
        elif self.llm_type == "ollama":
            return self._call_ollama(prompt)
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")

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

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama LLM."""
        try:
            response = self.llm_client.invoke(prompt)
            return response
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            raise

    def _parse_llm_response(self, response: str) -> dict[str, str]:
        """Parse JSON response from LLM."""
        # Remove markdown code blocks if present
        cleaned_response = re.sub(r"```json\s*|\s*```", "", response).strip()
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            return {"question": "Error parsing generation", "answer": "Error parsing generation"}
