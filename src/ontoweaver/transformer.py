""" The module that stores all the default transformers and the register functions.
"""
import re
import math
import sys
import json
import inspect
import logging
import pathlib
import importlib
from abc import abstractmethod

import numpy as np
import pandas as pd
import pandera.pandas as pa


from . import errormanager
from. import exceptions
from . import validate
from . import make_value
from . import make_labels
from . import base
from . import loader

logger = logging.getLogger("ontoweaver")


def register(transformer_class):
    """Adds the given transformer class to those available to OntoWeaver.

    The given class should inherit from ontoweaver.base.Transformer

    Example::

        import ontoweaver

        class user_transformer(ontoweaver.base.Transformer):
            def __init__(self, target, target_element_properties, edge=None, columns=None, **kwargs):
                super().__init__(target, target_element_properties, edge, columns, **kwargs)

            def __call__(self, row, i):
                for key in self.columns:
                    yield str(row[key])

        ontoweaver.transformer.register( user_transformer )

        # The mapping can now use "user_transformer" in the transformers list.

    Args:
        transformer_class: The class to add to the ontoweaver.transformer module.
    """

    if not issubclass(transformer_class, base.Transformer):
        logging.error(f"{transformer_class.__name__} should inherit from ontoweaver.base.Transformer.", section="transformer.register", exception = exceptions.InterfaceInheritanceError)
    current = sys.modules[__name__]
    logging.debug(f"Adding transformer {transformer_class.__name__}")
    setattr(current, transformer_class.__name__, transformer_class)


# NOTE: transformers pass all kwargs to superclass to allow it to show
#       the (additional) user-defined arguments when calling __repr__.


def import_from_path(file_path):
    """Import the given Python file path as a module."""
    # See https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    module_name = pathlib.Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def register_all(module_path):
    for mpath in module_path:
        logger.info(f"Look for transformers in `{mpath}`")
        mod = import_from_path(mpath)
        for name,cls in mod.__dict__.items():
            if inspect.isclass(cls):
                logger.debug(f"{cls}")
                if issubclass(cls, base.Transformer):
                    logger.info(f"    Register transformer: `{cls}`")
                    register(cls)


class map(base.Transformer):
    """Transformer subclass used for the simple mapping of cell values of defined columns and creating
    nodes with their respective values as id."""

    class ValueMaker(make_value.ValueMaker):
        def __init__(self, raise_errors: bool = True):
            super().__init__(raise_errors)

        def __call__(self, columns, row, i):
            for key in columns:
                if key not in row:
                    self.error(f"Column '{key}' not found in data", section="map.call",
                               exception=exceptions.TransformerDataError)
                else:
                    yield row[key]

    def __init__(self,
            properties_of,
            label_maker = None,
            branching_properties = None,
            columns=None,
            output_validator: validate.OutputValidator = None,
            multi_type_dict = None,
            raise_errors = True,
            **kwargs
        ):
        """
        Initialize the map transformer.

        Args:
            properties_of: Properties of the node.
            value_maker: the ValueMaker object used for the logic of cell value selection for each transformer.
            label_maker: the LabelMaker object used for handling the creation of the output of the transformer. Default is None.
            branching_properties: in case of branching on cell values, the dictionary holding the properties for each branch.
            columns: The columns to be processed.
            output_validator: the OutputValidator object used for validating transformer output.
            multi_type_dict: the dictionary holding regex patterns for node and edge type branching based on cell values.
            raise_errors: if True, the caller is asking for raising exceptions when an error occurs
        """

        self.value_maker = self.ValueMaker(raise_errors=raise_errors)

        super().__init__(properties_of,
            self.value_maker,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs
        )

        if not self.columns:
            self.error(f"No column declared for the `{type(self).__name__}` transformer, did you forgot to add a `columns` keyword?", section="map.call", exception = exceptions.TransformerInputError)


    def __call__(self, row, i):
        """
        Process a row and yield cell values as node IDs.

        Args:
            row: The current row of the DataFrame.
            i: The index of the current row.

        Yields:
            str: The cell value if valid.

        Raises:
            Warning: If the cell value is invalid.
        """
        for item in super().__call__(row, i):
            yield item


class split(base.Transformer):
    """Transformer subclass used to split cell values at defined separator and label_maker nodes with
    their respective values as id."""

    class ValueMaker(make_value.ValueMaker):

        def __init__(self, raise_errors: bool = True, separator: str = None):
            self.separator = separator
            super().__init__(raise_errors)

        def __call__(self, columns, row, i):
            for key in columns:
                if key not in row:
                    self.error(f"Column `{key}` not found in row:\n{row}",
                        "transformer.split",
                        exception = exceptions.TransformerConfigError)

                val = row[key]
                if isinstance(val, str):
                    items = val.split(self.separator)
                    for item in items:
                        yield item

                elif not base.is_not_null(val):
                    logger.debug(f"Value is null, I'll let my caller skip it.")
                    yield val  # Will be passed by super.__call__

                else:
                    try:  # Try generic access.
                        for item in val:
                            yield item
                    except Exception as e:
                        self.error(f"Cannot skip or iterate over {type(val)}: `{val}`. {e}",
                            "transformer.split",
                            exception = exceptions.TransformerDataError)

    def __init__(self,
        properties_of,
        label_maker = None,
        branching_properties = None,
        columns=None,
        output_validator: validate.OutputValidator = None,
        multi_type_dict = None,
        raise_errors = True,
        separator = None,
        **kwargs
    ):
        """
        Initialize the split transformer.

        Args:
            properties_of: Properties of the node.
            value_maker: the ValueMaker object used for the logic of cell value selection for each transformer.
            label_maker: the LabelMaker object used for handling the creation of the output of the transformer. Default is None.
            branching_properties: in case of branching on cell values, the dictionary holding the properties for each branch.
            columns: The columns to be processed.
            sep: Character(s) to use for splitting.
            output_validator: the OutputValidator object used for validating transformer output.
            raise_errors: if True, the caller is asking for raising exceptions when an error occurs
            separator: The character(s) to use for splitting the cell values. Defaults to ",".
        """

        assert columns, "I need at least 1 column to operate."
        assert isinstance(columns, list), "I need at least 1 column to operate."
        assert len(columns) >= 1, "I need at least 1 column to operate."

        self.separator = separator

        self.value_maker = self.ValueMaker(raise_errors=raise_errors, separator=self.separator)

        super().__init__(properties_of,
            self.value_maker,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict = multi_type_dict,
            raise_errors=raise_errors,
            **kwargs
        )


class cat(base.Transformer):
    """Transformer subclass used to concatenate cell values of defined columns and label_maker nodes with
    their respective values as id."""

    class ValueMaker(make_value.ValueMaker):
        def __init__(self, raise_errors: bool = True):
            super().__init__(raise_errors)

        def __call__(self, columns, row, i):
            formatted_items = ""
            for key in columns:
                formatted_items += str(row[key])

            yield formatted_items

    def __init__(self,
            properties_of,
            label_maker = None,
            branching_properties = None,
            columns=None,
            output_validator: validate.OutputValidator = None,
            multi_type_dict = None,
            raise_errors = True,
            **kwargs
        ):
        """
        Initialize the cat transformer.

        Args:
            properties_of: Properties of the node.
            value_maker: the ValueMaker object used for the logic of cell value selection for each transformer.
            label_maker: the LabelMaker object used for handling the creation of the output of the transformer. Default is None.
            branching_properties: in case of branching on cell values, the dictionary holding the properties for each branch.
            columns: The columns to be processed.
            output_validator: the OutputValidator object used for validating transformer output.
            raise_errors: if True, the caller is asking for raising exceptions when an error occurs
            multi_type_dict: the dictionary holding regex patterns for node and edge type branching based on cell values.
        """

        self.value_maker = self.ValueMaker(raise_errors=raise_errors)

        super().__init__(properties_of,
            self.value_maker,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs
        )

    def __call__(self, row, i):
        """
        Process a row and yield concatenated items as node IDs.

        Args:
            row: The current row of the DataFrame.
            i: The index of the current row.

        Yields:
            str: The concatenated string from the cell values.
        """
        if not self.columns:
            self.error(f"No column declared for the {type(self).__name__} transformer, did you forgot to add a `columns` keyword?", section=f"{type(self).__name__}.call", exception = exceptions.TransformerInputError)

        for item in super().__call__(row, i):
            yield item


class cat_format(base.Transformer):
    """Transformer subclass used to concatenate cell values of defined columns and label_maker nodes with
    their respective values as id."""

    class ValueMaker(make_value.ValueMaker):

        def __init__(self, raise_errors: bool = True, format_string: str = None):
            self.format_string = format_string
            super().__init__(raise_errors)

        def __call__(self, columns, row, i):

            try:
                formatted_string = self.format_string.format_map(row)
            except KeyError as err:
                self.error(f"{err}, available keys: {row}",
                    exception = exceptions.TransformerConfigError,
                    index = i,
                    section = "cat_format"
                )

            yield formatted_string

    def __init__(self,
            properties_of,
            label_maker = None,
            branching_properties = None,
            columns=None,
            output_validator: validate.OutputValidator = None,
            multi_type_dict = None,
            raise_errors = True,
            format_string = None,
             **kwargs
         ):
        """
        Initialize the cat_format transformer.

        Args:.
            target_element_properties: Properties of the node.
            value_maker: the ValueMaker object used for the logic of cell value selection for each transformer.
            label_maker: the LabelMaker object used for handling the creation of the output of the transformer. Default is None.
            branching_properties: in case of branching on cell values, the dictionary holding the properties for each branch.
            columns: The columns to be processed.
            format_string: A format string containing the column names to assemble.
            output_validator: the OutputValidator object used for validating transformer output.
            multi_type_dict: the dictionary holding regex patterns for node and edge type branching based on cell values.
            raise_errors: if True, the caller is asking for raising exceptions when an error occurs
        """

        self.value_maker = self.ValueMaker(
            raise_errors=raise_errors,
            format_string=format_string
        )

        super().__init__(properties_of,
            self.value_maker,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs
        )

        if not format_string:  # Neither empty string nor None.
            self.error(f"The `format_string` parameter of the `{type(self).__name__}` transformer cannot be an empty string.")



class rowIndex(base.Transformer):
    """Transformer subclass used for the simple mapping of nodes with row index values as id."""

    class ValueMaker(make_value.ValueMaker):
        def __init__(self, raise_errors: bool = True):
            super().__init__(raise_errors)

        def __call__(self, columns, row, i):
            yield i

    def __init__(self,
            properties_of,
            label_maker = None,
            branching_properties = None,
            columns=None,
            output_validator: validate.OutputValidator = None,
            multi_type_dict = None,
            raise_errors = True,
            **kwargs
        ):
        """
        Initialize the rowIndex transformer.

        Args:
            properties_of: Properties of the node.
            value_maker: the ValueMaker object used for the logic of cell value selection for each transformer.
            label_maker: the LabelMaker object used for handling the creation of the output of the transformer. Default is None.
            branching_properties: in case of branching on cell values, the dictionary holding the properties for each branch.
            columns: The columns to be processed.
            output_validator: the OutputValidator object used for validating transformer output.
            multi_type_dict: the dictionary holding regex patterns for node and edge type branching based on cell values.
            raise_errors: if True, the caller is asking for raising exceptions when an error occurs
        """

        self.value_maker = self.ValueMaker(raise_errors=raise_errors)

        super().__init__(properties_of,
            self.value_maker,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs
        )


class nested(base.Transformer):
    """Transformer subclass used for accessing a value within
       nested dictionaries or dataframes."""

    class ValueMaker(make_value.ValueMaker):
        def __init__(self, raise_errors: bool = True):
            super().__init__(raise_errors)

        def __call__(self, keys, dic, i):
            value = self.nested(keys, dic, i)
            if value:
                return [value]
            else:
                return []

        def nested(self, keys, dic, i, depth = " "):
            depth += "| "
            logger.debug(f"{depth}Received: {type(dic)}{keys}:\n{dic}")

            if dic is None:
                return None

            if isinstance(dic, str) and not keys:
                # Break recursivity.
                logger.debug(f"{depth}Ended with: `{dic}`")
                return dic

            if not isinstance(keys, str):
                logger.debug(f"{depth}I can iterate over keys")
                if isinstance(dic, str):
                    if ':' in dic and '{' in dic and '}' in dic:
                        # Probably a Python/JSON dictionary.
                        logger.debug(f"{depth}I'm parsing a JSON dictionary string.")
                        dic = json.loads(dic)
                        logger.debug(f"{depth}{dic}")
                    assert not isinstance(dic, str)

                # Consider it an object with bracket access, we just pass it.
                if isinstance(dic, np.ndarray):
                    if dic.shape == (0,):
                        return None
                    else:
                        return self.nested( keys[1:], dic[keys[0]], i, depth )

                elif keys[0] not in dic:
                    if isinstance(dic, dict):
                        available_keys = ', '.join(dic.keys())
                    elif isinstance(dic, pd.DataFrame):
                        available_keys = ', '.join(dic.columns)
                    else:
                        available_keys = f"[unknown object type `{type(dic)}`, I cannot read its keys or it does not have ones]"

                    msg = f"Key '{keys[0]}' not found in data, I can only see keys: {available_keys}."
                    msg += f" Object repr: {dic}."
                    self.error(msg, section="nested.call",
                               exception=exceptions.TransformerDataError)
                else:
                    # Recursive call until exhaustion.
                    logger.debug(f"{depth}Get: {type(dic)}[{keys[0]}]")
                    return self.nested( keys[1:], dic[keys[0]], i, depth )

    def __init__(self,
        properties_of,
        label_maker = None,
        branching_properties = None,
        columns=None,
        output_validator: validate.OutputValidator = None,
        multi_type_dict = None,
        raise_errors = True,
        **kwargs
    ):

        assert columns, "I need at least 1 key to operate."
        assert isinstance(columns, list), "I need at least 1 key to operate."
        assert len(columns) >= 1, "I need at least 1 key to operate."

        self.value_maker = self.ValueMaker(raise_errors=raise_errors)
        super().__init__(properties_of,
            self.value_maker,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs
        )

        self.keys = columns
        logger.debug(f"keys: {self.keys}")
        if not self.keys:
            self.error(f"No key declared for the `{type(self).__name__}` transformer, did you forgot to add a `keys` keyword?", section="nested.call", exception = exceptions.TransformerInputError)

    def __call__(self, row, i):
        for item in super().__call__(row, i):
            yield item


class split_nested(base.Transformer):

    def __init__(self,
        properties_of,
        label_maker = None,
        branching_properties = None,
        columns=None,
        output_validator: validate.OutputValidator = None,
        multi_type_dict = None,
        raise_errors = True,
        separator = None,
        **kwargs
    ):
        """
        FIXME doc
        """

        logger.debug(f"COLUMNS: {type(columns)}\n{columns}")
        assert columns, "I need 2 keys to operate."
        assert isinstance(columns, list), "I need several keys."
        assert len(columns) >= 2, "I need 2 keys, or you should use either split or nested."

        self.split = split(
            properties_of,
            label_maker,
            branching_properties,
            [columns[0]],
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            separator = separator,
            **kwargs,
        )

        keys = columns[1:]
        if not isinstance(keys, list):
            keys = [keys]

        self.nested = nested(
            properties_of,
            label_maker,
            branching_properties,
            keys,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs,
        )

        super().__init__(properties_of,
            self.split.value_maker,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs
        )

    def __call__(self, row, i):
        for rowval in self.split.value_maker(self.split.columns, row, i):
            val = self.nested.value_maker(self.nested.keys, rowval, i)

            value, edge_type, node_type, reverse_edge = self.create(val, row)
            if base.is_not_null(value):
                yield value, edge_type, node_type, reverse_edge


class capitalize(map):
    def __call__(self, row, i):
        """
        Process a row and yield cell values as node IDs,
        with first letter in uppercase.

        Args:
            row: The current row of the DataFrame.
            i: The index of the current row.

        Yields:
            str: The capitalized cell value if valid.

        Raises:
            Warning: If the cell value is invalid.
        """
        for item in super().__call__(row, i):
            value, edge_type, node_type, reverse_edge = item
            val = value.capitalize()
            yield val, edge_type, node_type, reverse_edge


class lower(map):
    def __call__(self, row, i):
        """
        Process a row and yield cell values as node IDs,
        with all letters in lowercase.

        Args:
            row: The current row of the DataFrame.
            i: The index of the current row.

        Yields:
            str: The capitalized cell value if valid.

        Raises:
            Warning: If the cell value is invalid.
        """
        for item in super().__call__(row, i):
            value, edge_type, node_type, reverse_edge = item
            val = value.lower()
            yield val, edge_type, node_type, reverse_edge


class upper(map):
    def __call__(self, row, i):
        """
        Process a row and yield cell values as node IDs,
        with all letters in uppercase.

        Args:
            row: The current row of the DataFrame.
            i: The index of the current row.

        Yields:
            str: The capitalized cell value if valid.

        Raises:
            Warning: If the cell value is invalid.
        """
        for item in super().__call__(row, i):
            value, edge_type, node_type, reverse_edge = item
            val = value.lower()
            yield val, edge_type, node_type, reverse_edge


class lower_capitalize(map):
    def __call__(self, row, i):
        """
        Process a row and yield cell values as node IDs,
        with first letter in uppercase, and all others in lowercase.

        Args:
            row: The current row of the DataFrame.
            i: The index of the current row.

        Yields:
            str: The capitalized cell value if valid.

        Raises:
            Warning: If the cell value is invalid.
        """
        for item in super().__call__(row, i):
            value, edge_type, node_type, reverse_edge = item
            val = value.lower().capitalize()
            yield val, edge_type, node_type, reverse_edge


class translate(base.Transformer):
    """Translate the targeted cell value using a tabular mapping and yield a node with using the translated ID."""

    class ValueMaker(make_value.ValueMaker):
        def __init__(self, translate, translate_from, translate_to, raise_errors: bool = True):
            self.translate = translate
            self.translate_from = translate_from
            self.translate_to = translate_to
            self.warn_lines = {}
            super().__init__(raise_errors)

        def __call__(self, columns, row, i):

            for key in columns:
                if key not in row:
                    self.error(f"Column '{key}' not found in data", section="translate",
                               exception = exceptions.TransformerDataError)
                cell = row[key]
                if cell in self.translate:
                    yield self.translate[cell]
                else:
                    msg = f"Some rows does not contain something to be translated"
                    f" from `{self.translate_from}` to `{self.translate_to}`"
                    f" at column `{key}`"
                    self.warn_lines.get(msg, []).append(i)

        def __del__(self):  # FIXME make warnings management a common interface
            if len(self.warn_lines) > 0:
                logging.warning(f"There was warnings in a translate transformer, on {len(self.warn_lines)} lines:")
                for w in self.warn_lines:
                    logging.warning(f"{w}: {', '.join(self.warn_lines[w])}")


    def __init__(self,
            properties_of,
            label_maker = None,
            branching_properties = None,
            columns=None,
            output_validator: validate.OutputValidator = None,
            multi_type_dict = None,
            raise_errors = True,
            **kwargs
        ):
        """
        Constructor.

        NOTE: The user should provide at least either `translations` or `translations_file`, but not both.

        Args:
            properties_of: Properties of the node.
            value_maker: the ValueMaker object used for the logic of cell value selection for each transformer.
            label_maker: the LabelMaker object used for handling the creation of the output of the transformer. Default is None.
            branching_properties: in case of branching on cell values, the dictionary holding the properties for each branch.
            columns: The columns to be processed.
            translations: A dictionary figuring what to replace (keys) with which string (values).
            translations_file: A filename pointing to a tabular file readable by Pandas' csv_read.
            translate_from: The column in the file containing what to replace.
            translate_to: The column in the file containing the replacement string.
            output_validator: the OutputValidator object used for validating transformer output.
            multi_type_dict: the dictionary holding regex patterns for node and edge type branching based on cell values.
            raise_errors: if True, the caller is asking for raising exceptions when an error occurs
            kwargs: Additional arguments to pass to a Loader function (e.g. if you want to load TSVs, "sep=TAB", reads the translations_file as tab-separated).
        """

        self.map = map(properties_of, label_maker, branching_properties, columns, output_validator, multi_type_dict, **kwargs)

        lpf = loader.LoadPandasFile()
        lpd = loader.LoadPandasDataframe()
        lrf = loader.LoadOWLFile()
        lrg = loader.LoadOWLGraph()

        # Since we cannot expand kwargs, let's recover what we have inside.
        self.translations = kwargs.get("translations", None)
        self.translations_file = kwargs.get("translations_file", None)
        self.translate_from = kwargs.get("translate_from", None)
        self.translate_to = kwargs.get("translate_to", None)

        if self.translations and self.translations_file:
            self.error(f"Cannot have both `translations` (=`{self.translations}`) and `translations_file` (=`{self.translations_file}`) defined in a {type(self).__name__} transformer.", section="translate", exception = exceptions.TransformerInterfaceError)

        if self.translations:
            self.translate = self.translations
            logger.debug(f"\t\t\tManual translations: `{self.translate}`")
        elif self.translations_file:
            logger.debug(f"\t\t\tGet translations from file: `{self.translations_file}`")
            if not self.translate_from:
                self.error(f"No translation source column declared for the `{type(self).__name__}` transformer using translations_file=`{self.translations_file}`, did you forget to add a `translate_from` keyword?", section="translate.init", exception = exceptions.TransformerInterfaceError)
            if not self.translate_to:
                self.error(f"No translation target column declared for the `{type(self).__name__}` transformer using translations_file=`{self.translations_file}`, did you forget to add a `translate_to` keyword?", section="translate.init", exception = exceptions.TransformerInterfaceError)
            else:
                # self.translations_file = translations_file
                # self.translate_from = translate_from
                # self.translate_to = translate_to

                # Possible arguments from the `translate` section.
                mapping_args = ["translations", "translations_file", "translate_from", "translate_to"]
                # Possible Python attributes.
                mapping_args += ["subclass"]
                # Discard match
                mapping_args += ["match"]
                # All possible arguments found in a YAML mapping.
                for attr in dir(base.MappingParser):
                    if re.match("^k_", attr):
                        mapping_args += getattr(base.MappingParser, attr)

                # Keep only the user-passed arguments that are not in possible YAML keywords.
                more_args = {k:v for k,v in kwargs.items() if k not in mapping_args}
                if "sep" in more_args:
                    if more_args['sep'] == 'TAB': # FIXME why the fuck is this changed somehow?
                        more_args['sep'] = '\t'

                logger.debug(f"\t\t\tAdditional user-passed arguments for the load function: {more_args}")

                self.df = pd.DataFrame()
                for with_loader in [lpf, lpd, lrf, lrg]:
                    if with_loader.allows([self.translations_file]):
                        logger.debug(f"\t\t\tUsing loader: {type(with_loader).__name__}")
                        try:
                            self.df = with_loader.load([self.translations_file], **more_args)
                        except exceptions.InputDataError as err:
                            logging.error(f"I cannot load the translations_file: `{self.translations_file}`")
                            raise err
                        break

                if self.df.empty:
                    self.error(f"I was not able to load a valid translations_file from: `{self.translations_file}`")

                logger.debug(f"Loaded a DataFrame: {self.df}")

                if self.translate_from not in self.df.columns:
                    self.error(f"Source column `{self.translate_from}` not found in {type(self).__name__} transformer’s translations file `{self.translations_file}`, available headers: `{','.join(self.df.columns)}`.", section="translate.init", exception = exceptions.TransformerDataError)

                if self.translate_to not in self.df.columns:
                    self.error(f"Target column `{self.translate_to}` not found in {type(self).__name__} transformer’s translations file `{self.translations_file}`, available headers: `{','.join(self.df.columns)}`.", section="translate.init", exception = exceptions.TransformerDataError)

                self.translate = {}
                for i,row in self.df.iterrows():
                    frm = row[self.translate_from]
                    to = row[self.translate_to]
                    if frm in self.translate and self.translate[frm] != to:
                        logger.warning(f"The key `{frm}` already exists in the translation table, and translated to `{self.translate[frm]}`. It now translates to `{to}`. You may want to avoid such duplicates in translation tables.")
                    if frm and to:
                        self.translate[frm] = to
                    else:
                        logger.warning(f"Cannot translate frm `{self.translate_from}` to `{self.translate_to}`, invalid translations values at row {i} of file `{self.translations_file}`: `{frm}` => `{to}`. I will ignore this translation.")

        else:
            self.error(f"When using a {type(self).__name__} transformer, you must define either `translations` or `translations_file`.", section="translate.init", exception = exceptions.TransformerInterfaceError)


        if not self.translate:
            self.error("No translation found, did you forget the `translations` keyword?", section="translate.init", exception = exceptions.TransformerInterfaceError)

        self.value_maker = self.ValueMaker(
            self.translate,
            self.translate_from,
            self.translate_to,
            raise_errors=raise_errors
        )

        super().__init__(properties_of,
            self.value_maker,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs
        )

    def __call__(self, row, i):
        """
        Process a row and yield cell values as node IDs.

        Args:
            row: The current row of the DataFrame.
            i: The index of the current row.

        Yields:
            str: The cell value if valid.

        Raises:
            Warning: If the cell value or the translation is invalid.
        """
        if not self.columns:
            self.error(f"No column declared for the {type(self).__name__} transformer, did you forgot to add a `columns` keyword?", section="translate", exception = exceptions.TransformerDataError)

        for item in super().__call__(row, i):
            yield item


class string(base.Transformer):
    """A transformer that makes up the given static string instead of extractsing something from the table."""

    class ValueMaker(make_value.ValueMaker):
        def __init__(self, raise_errors: bool = True, string: str = None):
            self.string = string
            super().__init__(raise_errors)

        def __call__(self, columns, row, i):
            yield self.string

    def __init__(self,
            properties_of,
            label_maker = None,
            branching_properties = None,
            columns=None,
            output_validator: validate.OutputValidator = None,
            multi_type_dict = None,
            raise_errors = True,
            **kwargs
        ):
        """
        Constructor.

        Args:
            properties_of: Properties of the node.
            value_maker: the ValueMaker object used for the logic of cell value selection for each transformer.
            label_maker: the LabelMaker object used for handling the creation of the output of the transformer. Default is None.
            branching_properties: in case of branching on cell values, the dictionary holding the properties for each branch.
            columns: The columns to be processed.
            value: The string to use.
            output_validator: the OutputValidator object used for validating transformer output.
            multi_type_dict: the dictionary holding regex patterns for node and edge type branching based on cell values.
            raise_errors: if True, the caller is asking for raising exceptions when an error occurs
        """

        self.value = kwargs.get("value", None)
        self.value_maker = self.ValueMaker(raise_errors=raise_errors, string=self.value)

        super().__init__(properties_of,
            self.value_maker,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs
        )

    def __call__(self, row, i):
        """
        Process a row and yield cell values as node IDs.

        Args:
            row: The current row of the DataFrame.
            i: The index of the current row.

        Yields:
            str: The cell value if valid.

        Raises:
            Warning: If the cell value is invalid.
        """
        if not self.value:
            self.error(f"No value passed to the {type(self).__name__} transformer, did you forgot to add a `value` keyword?", section="string.call", exception = exceptions.TransformerInterfaceError)

        for item in super().__call__(row, i):
            yield item


class replace(base.Transformer):
    """Transformer subclass used to remove characters that are not allowed from cell values of defined columns.
     The forbidden characters are defined by a regular expression pattern, and are substituted with a user-defined
     character or removed entirely. In case the cell value is made up of only forbidden characters, the node is not
     created and a warning is logged."""

    class ValueMaker(make_value.ValueMaker):

        def __init__(self, raise_errors: bool = True, forbidden = None, substitute = None):

            self.forbidden = forbidden
            self.substitute = substitute
            super().__init__(raise_errors)

        def __call__(self, columns, row, i):
            for key in columns:

                if not base.is_not_null(row[key]):
                    yield row[key]
                else: 
                    logger.debug(
                        f"Setting forbidden characters: {self.forbidden} for `replace` transformer, with substitute character: `{self.substitute}`.")
                    formatted = re.sub(self.forbidden, self.substitute, row[key])

                    strip_formatted = formatted.strip(self.substitute)
                    logger.debug(f"Formatted value: {strip_formatted}")
                    yield strip_formatted

    def __init__(self,
            properties_of,
            label_maker = None,
            branching_properties = None,
            columns=None,
            output_validator: validate.OutputValidator = None,
            multi_type_dict = None,
            raise_errors = True,
            **kwargs
        ):
        """
        Constructor.

        Args:
            properties_of: Properties of the node.
            value_maker: the ValueMaker object used for the logic of cell value selection for each transformer.
            label_maker: the LabelMaker object used for handling the creation of the output of the transformer. Default is None.
            branching_properties: in case of branching on cell values, the dictionary holding the properties for each branch.
            columns: The columns to be processed.
            forbidden: The regular expression pattern to match forbidden characters.
            substitute: The string to replace forbidden characters with.
            output_validator: the OutputValidator object used for validating transformer output.
            multi_type_dict: the dictionary holding regex patterns for node and edge type branching based on cell values.
            raise_errors: if True, the caller is asking for raising exceptions when an error occurs
        """
        self.forbidden = kwargs.get("forbidden", r'[^a-zA-Z0-9_`.()]') # By default, allow alphanumeric characters (A-Z, a-z, 0-9),
        # underscore (_), backtick (`), dot (.), and parentheses (). TODO: Add or remove rules as needed based on errors in Neo4j import.
        self.substitute = kwargs.get("substitute", "")

        self.value_maker = self.ValueMaker(raise_errors=raise_errors, forbidden=self.forbidden, substitute=self.substitute)

        super().__init__(properties_of,
            self.value_maker,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs
        )


class boolean(base.Transformer):
    """A transformer that can map any set of values onto a boolean pair.

    It considers a set of truth values, along with a set of falsehood values,
    and then set the node ID to the user's true or false value.

    If no configuration is given for ``consider_true`` and ``consider_false``,
    OntoWeaver will use Python's `bool(value)`` to assert the truth of the value
    passed from the cell.

    If ``output_true`` or ``output_false`` are omitted, they will default to "True"
    and "False".

    For instance:

    .. code:: yaml

        - boolean:
            column: my_column
            via_relation: my_relation
            consider_true:
                - Y
                - Yes
                - yes
            output_true: my_truth
            consider_false:
                - N
                - No
                - no
            output_false: my_falsehood

    Is equivalent to:

    .. code:: python

        if value in ["Y", "Yes", "yes"]:
            yield "my_truth"
        if value in ["N", "No", no"]:
            yield "my_falsehood"

    """

    class ValueMaker(make_value.ValueMaker):

        def __init__(self, raise_errors: bool = True, output_true = "True", output_false = "False", consider_true = None, consider_false = None):
            if (not consider_true and consider_false) or (consider_true and not consider_false):
                msg = "I can only handle both `consider_true` and `consider_false` being `None` at the same time. But here, one of them is `None` and the other is not."
                logger.error(msg)
                raise exceptions.TransformerConfigError(msg)

            elif consider_true and consider_false:
                if not isinstance(consider_true, list) or not isinstance(consider_false, list):
                    msg = "I can only consider both `consider_true` and `consider_false` being lists."
                    logger.error(msg)
                    raise exceptions.TransformerConfigError(msg)

                if len(consider_true) == 0 or len(consider_false) == 0:
                    msg = "I need both `consider_true` and `consider_false` to contain at least one value."
                    logger.error(msg)
                    raise exceptions.TransformerConfigError(msg)

                common = set(consider_true) & set(consider_false)
                if common:
                   msg = f"There are values that are common to both `consider_true` and `consider_false`: {common}, this makes no sense."
                   logger.error(msg)
                   raise exceptions.TransformerConfigError(msg)

            if output_true == output_false:
                msg = "Both `output_true` and `output_false` are the same value, this makes no sense."
                logger.error(msg)
                raise exceptions.TransformerConfigError(msg)

            if output_true is None or output_false is None:
                msg = "I need both `output_true` and `output_false` to have some value."
                logger.error(msg)
                raise exceptions.TransformerConfigError(msg)

            self.output_true = output_true
            self.output_false = output_false
            self.consider_false = consider_false
            self.consider_true = consider_true

            super().__init__(raise_errors)

        def __call__(self, columns, row, i):
            for key in columns:
                value = row[key]
                if not value:
                    continue
                value = str(value)

                if value in self.consider_true:
                    boo = True
                elif value in self.consider_false:
                    boo = False
                else:
                    logger.error(f"Value `{value}` is not found in either `consider_true` (`{'`, `'.join(self.consider_true)}`) or `consider_false` (`{'`, `'.join(self.consider_false)}`). I will bypass columns {columns} at row {i}.")
                    continue

                if boo:
                    out = self.output_true
                else:
                    out = self.output_false

                logger.debug(f"Made a boolean from `{value}` to `{out}`.")
                yield out

    def __init__(self,
             properties_of,
            label_maker = None,
            branching_properties = None,
            columns=None,
            output_validator: validate.OutputValidator = None,
            multi_type_dict = None,
            raise_errors = True,
            output_true = "True",
            output_false = "False",
            consider_true = None,
            consider_false = None,
            **kwargs
        ):

        self.value_maker = self.ValueMaker(
            raise_errors=raise_errors,
            output_true    = output_true,
            output_false   = output_false,
            consider_true  = [str(i) for i in consider_true],
            consider_false = [str(i) for i in consider_false],
        )

        super().__init__(properties_of,
            self.value_maker,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs
        )


class split_translate(base.Transformer):

    def __init__(self,
        properties_of,
        label_maker = None,
        branching_properties = None,
        columns=None,
        output_validator: validate.OutputValidator = None,
        multi_type_dict = None,
        raise_errors = True,
        separator = None,
        **kwargs
    ):
        """
        FIXME doc
        """

        self.split = split(
            properties_of,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            separator = separator,
            **kwargs,
        )

        self.translate = translate(
            properties_of,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs,
        )

        super().__init__(properties_of,
            self.split.value_maker,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs
        )

    def __call__(self, row, i):

        for value in self.split.value_maker(self.split.columns, row, i):
            if base.is_not_null(value):
                logging.debug(f"VALUE {value}")
                pseudorow = {"translate_column": value}
                for val in self.translate.value_maker(["translate_column"], pseudorow, i):
                    logging.debug(f"VAL {val}")
                    value, edge_type, node_type, reverse_edge = self.create(val, row)
                    yield value, edge_type, node_type, reverse_edge

class split_replace(base.Transformer):

    def __init__(self,
        properties_of,
        label_maker = None,
        branching_properties = None,
        columns=None,
        output_validator: validate.OutputValidator = None,
        multi_type_dict = None,
        raise_errors = True,
        separator = None,
        **kwargs
    ):
        """
        FIXME doc
        """

        self.split = split(
            properties_of,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            separator = separator,
            **kwargs,
        )

        self.replace = replace(
            properties_of,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs,
        )

        super().__init__(properties_of,
            self.split.value_maker,
            label_maker,
            branching_properties,
            columns,
            output_validator,
            multi_type_dict,
            raise_errors=raise_errors,
            **kwargs
        )

    def __call__(self, row, i):

        for value in self.split.value_maker(self.split.columns, row, i):
            if base.is_not_null(value):
                pseudorow = {"replace_column": value}
                for val in self.replace.value_maker(["replace_column"], pseudorow, i):
                    value, edge_type, node_type, reverse_edge = self.create(val, row)
                    yield value, edge_type, node_type, reverse_edge
