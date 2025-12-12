"""Define a base DataObject class that provides utilities for all other data object classes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional, TypeVar, overload

from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    from collections.abc import Sequence

TypeDataObject = TypeVar("TypeDataObject", bound="DataObject")


@dataclass
class DataObject:
    """Base class for serializable data objects.

    Provides JSON serialization/deserialization, dictionary conversion,
    and string templating. All subclasses are automatically registered
    in the class-level `data_object_registry`.

    Attributes
    ----------
    data_object_registry : dict[str, type[DataObject]]
        A registry mapping subclass names to their class types.

    """

    data_object_registry: ClassVar[dict[str, type[DataObject]]] = {}

    def __init_subclass__(cls) -> None:
        """Automatically register subclasses in the `data_object_registry`."""
        cls.data_object_registry[cls.__name__] = cls
        return super().__init_subclass__()

    @classmethod
    def save_list_to_json(cls, data_obj_lst: Sequence[DataObject], json_file_path: str) -> None:
        """Save a list of `DataObject` instances to a JSON file.

        :param data_obj_lst: List of `DataObject` instances to save.
        :type data_obj_lst: Sequence[DataObject]
        :param json_file_path: Path to the target JSON file.
        :type json_file_path: str
        """
        content = DataObject.convert_list_to_dict(list(data_obj_lst))
        with Path(json_file_path).open("w") as json_file:
            json.dump(content, json_file, indent=4)

    def save_to_json(self, json_file_path: str) -> None:
        """Save the `DataObject` instance to a JSON file.

        :param json_file_path: Path to the target JSON file.
        :type json_file_path: str
        """
        dict_data = self.convert_to_dict()
        with Path(json_file_path).open("w") as json_file:
            json.dump(dict_data, json_file, indent=4)

    @staticmethod
    def convert_list_to_dict(data_obj_inst: list[DataObject]) -> list[dict[str, Any]]:
        """Convert a list of `DataObject` instances to a list of dictionaries.

        :param data_obj_inst: The list `DataObject` instances to convert.
        :type data_obj_inst: list[DataObject].
        :return: A list of dictionaries representing the objects.
        :rtype: list[dict[str, Any]]
        """
        return [DataObject.get_dict_val(el) for el in data_obj_inst]

    def convert_to_dict(self) -> dict[str, Any]:
        """Convert the `DataObject` instance to a dictionary.

        :return: A dictionary representing the object.
        :rtype: dict[str, Any]
        """
        return {name: DataObject.get_dict_val(val) for name, val in self.__dict__.items()}

    @staticmethod
    def get_dict_val(val: Any) -> Any:  # noqa: ANN401
        """Convert an input val to a serializable val.

        :param val: The value to convert
        :type val: Any
        :return: The serializable version.
        :rtype: Any
        """
        match val:
            case list():
                return DataObject.convert_list_to_dict(val)
            case dict():
                return {key: DataObject.get_dict_val(val) for key, val in val.items()}
            case DataObject():
                return val.convert_to_dict()
            case _:
                return val

    @overload
    @classmethod
    def load_from_dict(
        cls: type[TypeDataObject],
        dict_content: list[dict[str, Any]],
        is_list: Optional[Literal[True]] = None,
    ) -> list[TypeDataObject]: ...

    @overload
    @classmethod
    def load_from_dict(
        cls: type[TypeDataObject],
        dict_content: dict[str, Any],
        is_list: Optional[Literal[False]] = None,
    ) -> TypeDataObject: ...

    @overload
    @classmethod
    def load_from_dict(
        cls: type[TypeDataObject],
        dict_content: dict[str, Any] | list[dict[str, Any]],
        is_list: Literal[None] = None,
    ) -> TypeDataObject | list[TypeDataObject]: ...
    @overload
    @classmethod
    def load_from_dict(
        cls: type[TypeDataObject],
        dict_content: dict[str, Any] | list[dict[str, Any]],
        is_list: Optional[bool] = None,
    ) -> TypeDataObject | list[TypeDataObject]: ...
    @classmethod
    def load_from_dict(
        cls: type[TypeDataObject],
        dict_content: dict[str, Any] | list[dict[str, Any]],
        is_list: Optional[bool] = None,
    ) -> TypeDataObject | list[TypeDataObject]:
        """Convert one or multiple dictionary dictionaries to one or multiple corresponding `DataObject` instances.

        :param cls: The class `DataObject` to convert the dictionaries to.
        :type cls: type[TypeDataObject]
        :param dict_content: The dictionaries to convert.
        :type dict_content: dict[str, Any] | list[dict[str,Any]]
        :param is_list: wether the input is a list of dictionaries if None the param is not used, defaults to None.
        :type is_list: Optional[bool], optional
        :raises TypeError: Raise a type error if the type of the dict_content is not coherent with the is_list param.
        :return: The data objects corresponding to the input dictionaries.
        :rtype: TypeDataObject | list[TypeDataObject]
        """
        if isinstance(dict_content, list) and (is_list or is_list is None):
            return [cls(**el) for el in dict_content]
        if isinstance(dict_content, dict) and (not is_list or is_list is None):
            return cls(**dict_content)
        raise TypeError

    @classmethod
    def get_json_content(cls, path_content: str) -> dict[str, Any] | list[dict[str, Any]]:
        """Get the dictionary corresponding to a json file.

        :param path_content: The path of the target json file.
        :type path_content: str
        :return: The dictionary corresponding to the json file
        :rtype: dict[str, Any]
        """
        with Path(path_content).open("r") as json_docs:
            content = json.load(json_docs)
        return content

    @overload
    @classmethod
    def load_from_json(
        cls: type[TypeDataObject],
        json_file_path: str,
        is_list: Literal[True],
    ) -> list[TypeDataObject]: ...

    @overload
    @classmethod
    def load_from_json(cls: type[TypeDataObject], json_file_path: str, is_list: Literal[False]) -> TypeDataObject: ...

    @overload
    @classmethod
    def load_from_json(
        cls: type[TypeDataObject],
        json_file_path: str,
        is_list: Literal[None] = None,
    ) -> TypeDataObject | list[TypeDataObject]: ...

    @classmethod
    def load_from_json(
        cls: type[TypeDataObject],
        json_file_path: str,
        is_list: Optional[bool] = None,
    ) -> TypeDataObject | list[TypeDataObject]:
        """Get one or many `DataObject` instances from a json file.

        :param cls: The `DataObject` class to convert the content of the json file to.
        :type cls: type[TypeDataObject]
        :param json_file_path: The target json file path.
        :type json_file_path: str
        :param is_list: Wether the content of the json is a list, if None the content is unknown
        and the return type can be one or many data objects, defaults to None.
        :type is_list: Optional[bool], optional
        :return: The result `DataObject` instance corresponding to json file.
        :rtype:  TypeDataObject | list[TypeDataObject]:
        """
        json_content = cls.get_json_content(json_file_path)
        return cls.load_from_dict(json_content, is_list)
