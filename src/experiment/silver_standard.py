import json
import logging
import random
import re
import uuid
from typing import Any

# Setup logger
logger = logging.getLogger(__name__)


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
        self.model = "gemini-3-pro-preview"
        self.documents = documents
        logger.debug(
            f"SilverStandardGenerator initialized with {len(documents)} documents and model={self.model}"
        )

    def generate_dataset(self, num_samples: int, num_hops: int = 3) -> list[dict[str, Any]]:
        """Generate a dataset of multi-hop QA pairs."""
        logger.info(
            f"Starting dataset generation: target={num_samples} samples, num_hops={num_hops}"
        )
        dataset = []
        failures = 0
        max_failures = num_samples * 10

        while len(dataset) < num_samples:
            sample = self.generate_sample(num_hops=num_hops)
            if sample:
                dataset.append(sample)
                logger.debug(
                    f"Generated sample {len(dataset)}/{num_samples}: {sample['question'][:60]}..."
                )
            else:
                failures += 1
                if failures % 5 == 0:  # Log every 5 failures
                    logger.debug(
                        f"Failure rate: {failures} failures, {len(dataset)}/{num_samples} successful samples"
                    )
                if failures >= max_failures:
                    logger.warning(
                        f"High failure rate exceeded. Stopped after {failures} failures. "
                        f"Generated {len(dataset)}/{num_samples} samples."
                    )
                    print(
                        f"⚠ Warning: High failure rate. Stopped after {failures} failures. "
                        f"Generated {len(dataset)}/{num_samples} samples."
                    )
                    break

        logger.info(
            f"Dataset generation complete: {len(dataset)}/{num_samples} samples generated successfully"
        )
        return dataset

    def generate_sample(self, num_hops: int = 3) -> dict[str, Any] | None:
        """Generate a single multi-hop QA pair with evidence snippets."""
        try:
            chunks = self._get_random_contexts(n=num_hops)
            logger.debug(f"Selected {len(chunks)} contexts for sample generation")

            # Build and send prompt
            prompt = self._build_multihop_prompt(chunks)
            response_text = self._call_llm(prompt)
            qa_data = self._parse_llm_response(response_text)

            question = qa_data.get("question", "")
            answer = qa_data.get("answer", "")
            gold_snippets = qa_data.get("gold_snippets", [])

            # Validate response
            if question == "IMPOSSIBLE":
                logger.debug("LLM determined question is IMPOSSIBLE (no valid bridge entity)")
                return None
            if answer == "IMPOSSIBLE":
                logger.debug("LLM determined answer is IMPOSSIBLE")
                return None
            if question == "Error parsing generation":
                logger.warning("Failed to parse LLM response as JSON")
                return None
            if not gold_snippets:
                logger.debug("LLM response missing gold_snippets")
                return None

            sample = {
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
            logger.debug(f"Sample generated successfully - Q: '{question[:50]}...' / A: '{answer}'")
            return sample
        except Exception as e:
            logger.error(f"Unexpected error generating sample: {e}", exc_info=True)
            return None

    def _get_random_contexts(self, n: int = 3, min_char_length: int = 100) -> list[str]:
        candidates = [doc for doc in self.documents if len(doc) >= min_char_length]
        if len(candidates) < n:
            logger.warning(
                f"Requested {n} contexts but only {len(candidates)} documents meet min_char_length={min_char_length}"
            )
        selected = random.sample(candidates, n) if len(candidates) >= n else candidates
        logger.debug(
            f"Selected {len(selected)} contexts (min_length={min_char_length}, available={len(candidates)}/{len(self.documents)})"
        )
        return selected

    def _build_multihop_prompt(self, contexts: list[str]) -> str:
        """Build the prompt for multi-hop question generation with strict bridge validation."""
        context_text = "\n\n".join(
            [f"Context {i + 1}: {chunk}" for i, chunk in enumerate(contexts)]
        )

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
            logger.debug(f"Calling Gemini API with model={self.model}, prompt_length={len(prompt)}")
            response = self.llm_client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            logger.debug(f"Gemini API response received: {len(response.text)} chars")
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            print(f"Error calling Gemini API: {e}")
            raise

    def _parse_llm_response(self, response: str) -> dict[str, str]:
        """Parse JSON response from LLM."""
        cleaned_response = re.sub(r"```json\s*|\s*```", "", response).strip()
        try:
            parsed = json.loads(cleaned_response)
            logger.debug(
                f"Successfully parsed LLM response: {len(cleaned_response)} chars -> {list(parsed.keys())}"
            )
            return parsed
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse LLM response as JSON: {e}\nResponse preview: {cleaned_response[:200]}..."
            )
            return {"question": "Error parsing generation", "answer": "Error parsing generation"}
