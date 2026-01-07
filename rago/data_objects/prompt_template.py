"""Define the prompt template data class."""

from __future__ import annotations

import string

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """A Prompt containing a string to be filled.

    :param content: the text content of the prompt
    :type content: str
    """

    content: str = Field(..., min_length=1, strict=True)

    def get_format_params(self) -> list[str]:
        """Get the params to set in the prompt.

        :return: the list of params to set in the prompt.
        :rtype: list[str]
        """
        formatter = string.Formatter()
        return [fname for _, fname, _, _ in formatter.parse(self.content) if fname]

    def get_partially_formatted_prompt_template(self, **kwargs: str) -> PromptTemplate:
        """Get a partially formatted prompt template.

        :param kwargs: The args to set in the new prompt Template
        :type kwargs: str
        :return: The new prompt template we some field fixed already.
        :rtype: str
        """
        list_params = self.get_format_params()
        params_to_keep = {param: "{" + param + "}" for param in list_params if param not in kwargs}
        kwargs = params_to_keep | kwargs
        new_content = self.content.format(**kwargs)
        return PromptTemplate(new_content)

    def get_filled_prompt(self, sep: str = "\n", **kwargs: str | list[str] | None) -> str:
        """Get the prompt filled with the keyword arguments.

        :param sep: The separator to put in between strings of a list of string args.
        :type sep: str
        :param kwargs: The params to add to the prompt.
        :type kwargs: str | list[str]
        :return: the filled prompt
        :rtype: str
        """
        preprocessed_kwargs = self.preprocess_kwargs(sep, **kwargs)
        return self.content.format(**preprocessed_kwargs)

    def preprocess_kwargs(self, sep: str = "\n", **kwargs: str | list[str] | None) -> dict[str, str]:
        """Preprocess the keyword args to convert them to string if needed.

        :param sep: The separator to put in between strings of a list of string args.
        :type sep: str
        :param kwargs: The params to preprocess.
        :type kwargs: str | list[str]
        """
        processed_kwargs = {}
        for key_arg, val_arg in kwargs.items():
            match val_arg:
                case str():
                    processed_kwargs[key_arg] = val_arg
                case list():
                    processed_kwargs[key_arg] = self.aggregate_strings(val_arg, sep)
                case _:
                    continue
        return processed_kwargs

    def aggregate_strings(self, list_strings: list[str], separator: str) -> str:
        """Aggregate a list of strings.

        :param list_strings: List of strings to aggregate.
        :type list_strings: list[str]
        :param separator: the separation symbol in between the aggregated strings.
        :type separator: str
        """
        return separator.join(list_strings)
