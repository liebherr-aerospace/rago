"""Define a simple dataset generator that generate a fixed number of questions per document."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from rago.data_objects import Document, EvalSample
from rago.dataset.generator import BaseDatasetGenerator
from rago.dataset.rag_dataset import RAGDataset
from rago.prompts import DEFAULT_DATASET_GENERATION_PROMPT

if TYPE_CHECKING:
    from rago.model.wrapper.llm_agent import LLMAgent

SeedDataType = RAGDataset | list[Document]


class SimpleDatasetGenerator(BaseDatasetGenerator[SeedDataType, RAGDataset]):
    """A simple BaseDatasetGenerator that generates a fixed number of questions per document.

    In RAGO, it is used by the Optimization manager to generate a dataset to test RAG configurations on.
    """

    def __init__(
        self,
        generator: Optional[LLMAgent] = None,
        generation_prompt: str = DEFAULT_DATASET_GENERATION_PROMPT,
        number_questions_per_document: int = 1,
    ) -> None:
        """Instantiate a simple dataset generator that uses a llm to generate a fixed number of questions per document.

        :param generator: Language model used to generate the dataset.
        :type generator: LanguageModel
        :param generation_prompt: The prompt used by the generator to generate eval samples.
        :type generation_prompt: str
        :param number_questions_per_document: The number of question to generate per document.
        :type number_questions_per_document: int
        """
        super().__init__(generator, generation_prompt)
        self.number_questions_per_document = number_questions_per_document

    def generate_dataset(self, seed_data: Optional[SeedDataType] = None) -> RAGDataset:
        """Generate the dataset from a corpus with a fixed number of questions per document.

        :param seed_data: data containing the corpus used to generate question, defaults to None.
        :type seed_data: Optional[SeedDataType], optional
        :raises ValueError: The seed_data is not set.
        :return: The generated synthetic dataset.
        :rtype: RAGDataset
        """
        if seed_data is None:
            raise ValueError
        samples: list[EvalSample] = seed_data.samples.copy() if isinstance(seed_data, RAGDataset) else []
        corpus = seed_data.corpus if isinstance(seed_data, RAGDataset) else {str(uuid4()): doc for doc in seed_data}
        for document in corpus.values():
            new_samples = self.generate_eval_samples_from_documents(document)
            samples += new_samples

        return RAGDataset(samples=samples, corpus=corpus)

    def generate_eval_samples_from_documents(self, document: Document) -> list[EvalSample]:
        """Generate eval samples from the documents.

        The number of questions generated is equal to the attribute number_questions_per_document.
        What the function effectively does is ask the model to generate k questions.
        If the model generates less questions than asked
        it will be iteratively called until all the questions are generated.
        The eventual surplus questions are trimmed to only return number_questions_per_document.
        :param document: The document to generate questions on.
        :type document: Document
        :return: The list of generated eval samples for the document (i.e queries-context pairs).
        :rtype: list[str]
        """
        num_question_left_to_generate = self.number_questions_per_document
        questions: list[EvalSample] = []
        while num_question_left_to_generate > 0:
            prompts = self.generation_prompt.get_filled_prompt(
                documents=document.text,
                number_of_question=str(num_question_left_to_generate),
            )
            generation = self.generator.query(prompts)
            generated_question = self.parse_generation(generation, document)
            num_question_left_to_generate -= len(generated_question)
            questions += generated_question
        return questions[: self.number_questions_per_document]

    def parse_generation(self, generation: str, document: Document) -> list[EvalSample]:
        """Parse the output of the generator model to get a new examples.

        In this simple implementation the parsing return all non empty line as questions.

        :parameter generation: The generation output from the generator to parse.
        :type generation: str
        :param document: The document the questions are generated on.
        :type document: Document
        :return: The new test samples that contains the generated questions.
        :rtype: EvalSample
        """
        blank_line_regex = r"\n+"
        questions = re.split(blank_line_regex, generation.strip())
        questions = [q for q in questions if len(q) > 0]
        return [EvalSample(query=question, context=[document]) for question in questions]
