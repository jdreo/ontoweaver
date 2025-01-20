import sys
import math
import threading
import types as pytypes
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from collections.abc import Iterable
from enum import Enum, EnumMeta
import yaml

import pandas as pd
import pandera as pa
from numpy.ma.core import set_fill_value

from . import base
from . import types
from . import transformer
from . import exceptions
from . import validate

class MetaEnum(EnumMeta):
    """
    Metaclass for Enum to allow checking if an item is in the Enum.
    """

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class Enumerable(Enum, metaclass=MetaEnum):
    """
    Base class for Enums with MetaEnum metaclass.
    """
    pass


class TypeAffixes(str, Enumerable):
    """
    Enum for type affixes used in ID creation.
    """
    suffix = "suffix"
    prefix = "prefix"
    none = "none"


class PandasAdapter(base.Adapter):
    """Interface for extracting data from a Pandas DataFrame with a simple mapping configuration based on declared types.

    The general idea is that each row of the table is mapped to a source node,
    and some column values are mapped to an edge leading to another node.
    Some other columns may also be mapped to properties of either a node or an edge.

    The class expect a configuration formed by three objects:
        - the type of the source node mapped for each row.
        - a dictionary mapping each column name to the type of the edge (which contains the type of both the source and target node),
        - a dictionary mapping each (node or edge) type to another dictionary listing which column is extracted to which property.

    Note that, when using the `configure` mapping,
    types are created by default in the `ontoweaver.types` module,
    so that you may access the list of all declared types by using:
        - `ontoweaver.types.all.nodes()`,
        - `ontoweaver.types.all.node_fields()`,
        - `ontoweaver.types.all.edges()`,
        - `ontoweaver.types.all.edge_fields()`.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 subject_transformer: base.Transformer,
                 transformers: Iterable[base.Transformer],
                 metadata: Optional[dict] = None,
                 validator: Optional[validate.InputValidator] = None,
                 type_affix: Optional[TypeAffixes] = TypeAffixes.suffix,
                 type_affix_sep: Optional[str] = ":",
                 parallel_mapping: int = 0,
                 raise_errors = True,
                 ):
        """
        Instantiate the adapter.

        Args:
            df (pd.DataFrame): The table containing the input data.
            subject_transformer (base.Transformer): The transformer that maps the subject node.
            transformers (Iterable[base.Transformer]): List of transformer instances that map the data frame to nodes and edges.
            metadata (Optional[dict]): Metadata to be added to all the nodes and edges.
            type_affix (Optional[TypeAffixes]): Where to add a type annotation to the labels (either TypeAffixes.prefix, TypeAffixes.suffix or TypeAffixes.none).
            type_affix_sep (Optional[str]): String used to separate a label from the type annotation (WARNING: double-check that your BioCypher config does not use the same character as a separator).
            parallel_mapping (int): Number of workers to use in parallel mapping. Defaults to 0 for sequential processing.
            raise_errors (bool): if True, will raise an exception when an error is encountered, else, will log the error and try to proceed.
        """
        super().__init__(raise_errors)

        logging.info("DataFrame info:")
        # logging.info(df.info())
        logging.debug("Columns:")
        for c in df.columns:
            logging.debug(f"\t`{c}`")
        logging.info("\n" + str(df))
        self.df = df

        self.validator = validator

        if not type_affix in TypeAffixes:
            raise ValueError(f"`type_affix`={type_affix} is not one of the allowed values ({[t for t in TypeAffixes]})")
        else:
            self.type_affix = type_affix

        self.type_affix_sep = type_affix_sep

        self.subject_transformer = subject_transformer
        self.transformers = transformers
        self.metadata = metadata
        self.schema = validator
        # logging.debug(self.properties_of)
        self.parallel_mapping = parallel_mapping

    def source_type(self, row):
        """
        Accessor to the row type actually used by `run`.

        You may overload this function if you want
        to make the row type dependent on some column value.

        By default, just return the default row type defined in the constructor,
        without taking the row values into account.

        Args:
            row: The current row of the DataFrame.

        Returns:
            The row type.
        """
        return self.row_type


    def make_id(self, entry_type, entry_name):
        """
        Create a unique id for the given cell consisting of the entry name and type,
        taking into account affix and separator configuration.

        Args:
            entry_type: The type of the entry.
            entry_name: The name of the entry.

        Returns:
            str: The created ID.

        Raises:
            ValueError: If the ID creation fails.
        """
        assert(type(entry_type) == str)
        if type(entry_name) != str:
            logging.warning(f"Identifier '{entry_name}' (of type '{entry_type}') is not a string, I had to convert it explicitely, check that the related transformer yields a string.")
            entry_name = str(entry_name)

        if self.type_affix == TypeAffixes.prefix:
            id = f'{entry_type}{self.type_affix_sep}{entry_name}'
        elif self.type_affix == TypeAffixes.suffix:
            id = f'{entry_name}{self.type_affix_sep}{entry_type}'
        elif self.type_affix == TypeAffixes.none:
            id = f'{entry_name}'

        if id:
            logging.debug(f"\t\tFormatted ID `{id}` for cell value `{entry_name}` of type: `{entry_type}`")
            return id
        else:
            self.error(f"Failed to format ID for cell value: `{entry_name}` of type: `{entry_type}`", exception = exceptions.DeclarationError)


    def valid(self, val):
        """
        Checks if cell value is valid - not a `nan`.

        Args:
            val: The value to check.

        Returns:
            bool: True if the value is valid, False otherwise.
        """
        if pd.api.types.is_numeric_dtype(type(val)):
            if (math.isnan(val) or val == float("nan")):
                return False
        elif str(val) == "nan":  # Conversion from Pandas' `object` needs to be explicit.
            return False
        return True


    def properties(self, properity_dict, row, i, transformer, node = False):

        """
        Extract properties of each property category for the given node type.
        If no properties are found, return an empty dictionary.

        Args:
            properity_dict: Dictionary of property mappings.
            row: The current row of the DataFrame.
            i: The index of the current row.
            transformer: The transformer instance creating the node or edge class at hand.
            node: True if the object created is a node, False otherwise.

        Returns:
            dict: Extracted properties.
        """
        properties = {}

        for prop_transformer, property_name in properity_dict.items():
            for property in prop_transformer(row, i):
                properties[property_name] = str(property).replace("'", "`")

        # If the metadata dictionary is not empty add the metadata to the property dictionary.
        if self.metadata:
            if node:
                elem = transformer.target
            else:
                elem = transformer.edge
            if elem.__name__ in self.metadata:
                for key, value in self.metadata[elem.__name__].items():
                    properties[key] = value

        return properties


    def make_node(self, node_t, id, properties):
        """
        Create nodes of a certain type.

        Args:
            node_t: The type of the node.
            id: The ID of the node.
            properties: The properties of the node.

        Returns:
            The created node.
        """
        return node_t(id=id, properties=properties)


    def make_edge(self, edge_t, id_target, id_source, properties):
        """
        Create edges of a certain type.

        Args:
            edge_t: The type of the edge.
            id_target: The ID of the target node.
            id_source: The ID of the source node.
            properties: The properties of the edge.

        Returns:
            The created edge.
        """
        return edge_t(id_source=id_source, id_target=id_target, properties=properties)

    def run(self):
        """Iterate through dataframe in parallel and map cell values according to YAML file, using a list of transformers."""

        # Thread-safe containers with their respective locks
        self._nodes = []
        self._edges = []
        self._errors = []
        self._nodes_lock = threading.Lock()
        self._edges_lock = threading.Lock()
        self._errors_lock = threading.Lock()
        self._row_lock = threading.Lock()
        self._transformations_lock = threading.Lock()
        self._local_nb_nodes_lock = threading.Lock()

        # Function to process a single row and collect operations
        def process_row(row_data):
            i, row = row_data
            local_nodes = []
            local_edges = []
            local_errors = []
            local_rows = 0
            local_transformations = 0
            local_nb_nodes = 0

            logging.debug(f"Process row {i}...")
            local_rows += 1
            # There can be only one subject, so transformers yielding multiple IDs cannot be used.
            logging.debug("\tCreate subject node:")
            ids = list(self.subject_transformer(row, i))
            if (len(ids) > 1):
                local_errors.append(self.error(f"You cannot use a transformer yielding multiple IDs as a subject. Subject Transformer `{self.subject_transformer}` produced multiple IDs: {ids}", indent=2, exception = exceptions.TransformerInterfaceError))
            source_id = ids[0]
            source_node_id = self.make_id(self.subject_transformer.target.__name__, source_id)

            if source_node_id:
                logging.debug(f"\t\tDeclared subject ID: {source_node_id}")
                local_nodes.append(self.make_node(node_t=self.subject_transformer.target, id=source_node_id,
                                                  properties=self.properties(self.subject_transformer.properties_of,
                                                                             row, i, self.subject_transformer,
                                                                             node=True)))
            else:
                local_errors.append(self.error(f"Failed to declare subject ID for row #{i}: `{row}`.", indent=2, exception = exceptions.DeclarationError))

            # Loop over list of transformer instances and create corresponding nodes and edges.
            # FIXME the transformer variable here shadows the transformer module.
            for j,transformer in enumerate(self.transformers):
                local_transformations += 1
                logging.debug(f"\tCalling transformer: {transformer}...")
                for target_id in transformer(row, i):
                    local_nb_nodes += 1
                    if target_id:
                        target_node_id = self.make_id(transformer.target.__name__, target_id)
                        logging.debug(f"\t\tMake node {target_node_id}")
                        local_nodes.append(self.make_node(node_t=transformer.target, id=target_node_id,
                                                          properties=self.properties(transformer.properties_of, row,
                                                                                     i, transformer, node=True)))

                        # If a `from_subject` attribute is present in the transformer, loop over the transformer
                        # list to find the transformer instance mapping to the correct type, and then create new
                        # subject id.

                        # FIXME add hook functions to be overloaded.

                        # FIXME: Make from_subject reference a list of subjects instead of using the add_edge function.

                        if hasattr(transformer, "from_subject"):

                            found_valid_subject = False

                            for t in self.transformers:
                                if transformer.from_subject == t.target.__name__:
                                    found_valid_subject = True
                                    for s_id in t(row, i):
                                        subject_id = s_id
                                        subject_node_id = self.make_id(t.target.__name__, subject_id)
                                        logging.debug(
                                            f"\t\tMake edge from {subject_node_id} toward {target_node_id}")
                                        local_edges.append(
                                            self.make_edge(edge_t=transformer.edge, id_source=subject_node_id,
                                                           id_target=target_node_id,
                                                           properties=self.properties(transformer.properties_of,
                                                                                      row, i, t)))

                                else:
                                    continue

                            if not found_valid_subject:
                                local_errors.append(self.error(f"\t\t\tInvalid subject declared from {transformer}."
                                                               f" The subject you declared in the `from_subject` directive: `{transformer.from_subject}` must not be the same as the default subject type.",
                                                               exception=exceptions.ConfigError))


                        else: # no attribute `from_subject` in `transformer`
                            logging.debug(f"\t\tMake edge from {source_node_id} toward {target_node_id}")
                            local_edges.append(self.make_edge(edge_t=transformer.edge, id_target=target_node_id,
                                                              id_source=source_node_id,
                                                              properties=self.properties(transformer.edge.fields(),
                                                                                         row, i, transformer)))
                    else:
                        local_errors.append(self.error(f"No valid target node identifier from {transformer} for {i}th row.", indent=2, section="transformers", index=j, exception = exceptions.TransformerDataError))
                        continue

            return local_nodes, local_edges, local_errors, local_rows, local_transformations, local_nb_nodes
        # End of process_row local function

        nb_rows = 0
        nb_transformations = 0
        nb_nodes = 0

        if self.parallel_mapping > 0:
            logging.info(f"Processing dataframe in parallel. Number of workers set to: {self.parallel_mapping} ...")
            # Process the dataset in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                # Map the process_row function across the dataframe
                results = list(executor.map(process_row, self.df.iterrows()))

            # Append the results in a thread-safe manner after all rows have been processed
            for local_nodes, local_edges, local_errors, local_rows, local_transformations, local_nb_nodes in results:
                with self._nodes_lock:
                    self.nodes_append(local_nodes)
                with self._edges_lock:
                    self.edges_append(local_edges)
                with self._errors_lock:
                    self.errors += local_errors
                with self._row_lock:
                    nb_rows += local_rows
                with self._transformations_lock:
                    nb_transformations += local_transformations
                with self._local_nb_nodes_lock:
                    nb_nodes += local_nb_nodes

        elif self.parallel_mapping == 0:
            logging.info(f"Processing dataframe sequentially...")
            for i, row in self.df.iterrows():
                local_nodes, local_edges, local_errors, local_rows, local_transformations, local_nb_nodes = process_row((i, row))
                self.nodes_append(local_nodes)
                self.edges_append(local_edges)
                self.errors += local_errors
                nb_rows += local_rows
                nb_transformations += local_transformations
                nb_nodes += local_nb_nodes

        else:
            self.error(f"Invalid value for `parallel_mapping` ({self.parallel_mapping})."
                       f" Pass 0 for sequential processing, or the number of workers for parallel processing.", exception = exceptions.ConfigError)

        # Final logging
        if self.errors:
            logging.error(
                f"Recorded {len(self.errors)} errors while processing {nb_transformations} transformations with {len(self.transformers)} transformers, producing {nb_nodes} nodes for {nb_rows} rows.")
            # logging.debug("\n".join(self.errors))
        else:
            logging.info(
                f"Performed {nb_transformations} transformations with {len(self.transformers)} transformers, producing {nb_nodes} nodes for {nb_rows} rows.")


def extract_all(df: pd.DataFrame, config: dict, parallel_mapping = 0, module = types, affix = "suffix", separator = ":"):
    """
    Proxy function for extracting from a table all nodes, edges and properties
    that are defined in a PandasAdapter configuration.

    Args:
        df (pd.DataFrame): The DataFrame containing the input data.
        config (dict): The configuration dictionary.
        parallel_mapping (int): Number of workers to use in parallel mapping. Defaults to 0 for sequential processing.
        module: The module in which to insert the types declared by the configuration.
        affix (str): The type affix to use (default is "suffix").
        separator (str): The separator to use between labels and type annotations (default is ":").

    Returns:
        PandasAdapter: The configured adapter.
    """
    parser = YamlParser(config, module)
    mapping = parser()

    adapter = PandasAdapter(
        df,
        *mapping,
        type_affix=affix,
        type_affix_sep=separator,
        parallel_mapping=parallel_mapping,
    )

    adapter.run()

    return adapter



class Declare(base.ErrorManager):
    """
    Declarations of functions used to declare and instantiate object classes used by the Adapter for the mapping
    of the data frame.

    Args:
        module: The module in which to insert the types declared by the configuration.
    """

    def __init__(self,
                 module=types,
                 raise_errors = True,
                 ):
        super().__init__(raise_errors)
        self.module = module

    def make_node_class(self, name, properties={}, base=base.Node):
        """
        Create a node class with the given name and properties.

        Args:
            name: The name of the node class.
            properties (dict): The properties of the node class.
            base: The base class for the node class.

        Returns:
            The created node class.
        """
        # If type already exists, return it.
        if hasattr(self.module, name):
            cls = getattr(self.module, name)
            logging.debug(
                f"\t\tNode class `{name}` (prop: `{cls.fields()}`) already exists, I will not create another one.")
            for p in properties.values():
                if p not in cls.fields():
                    logging.warning(f"\t\t\tProperty `{p}` not found in declared fields for node class `{cls.__name__}`.")
            return cls

        def fields():
            return list(properties.values())

        attrs = {
            "__module__": self.module.__name__,
            "fields": staticmethod(fields),
        }
        t = pytypes.new_class(name, (base,), {}, lambda ns: ns.update(attrs))
        logging.debug(f"\t\tDeclare Node class `{t.__name__}` (prop: `{properties}`).")
        setattr(self.module, t.__name__, t)
        return t

    def make_edge_class(self, name, source_t, target_t, properties={}, base=base.Edge, ):
        """
        Create an edge class with the given name, source type, target type, and properties.

        Args:
            name: The name of the edge class.
            source_t: The source type of the edge.
            target_t: The target type of the edge.
            properties (dict): The properties of the edge class.
            base: The base class for the edge class.

        Returns:
            The created edge class.
        """
        # If type already exists, return it.
        if hasattr(self.module, name):
            cls = getattr(self.module, name)
            logging.info(
                f"\t\tEdge class `{name}` (prop: `{cls.fields()}`) already exists, I will not create another one.")
            for t, p in properties.items():
                if p not in cls.fields():
                    logging.warning(f"\t\tProperty `{p}` not found in fields.")

            tt_list = cls.target_type()

            # FIXME: should we keep target_type approach?
            tt_list.append(target_t)

            def tt():
                return tt_list

            cls.target_type = staticmethod(tt)

            return cls

        def fields():
            return properties

        def st():
            return source_t

        def tt():
            return [target_t]

        attrs = {
            "__module__": self.module.__name__,
            "fields": staticmethod(fields),
            "source_type": staticmethod(st),
            "target_type": staticmethod(tt),
        }
        t = pytypes.new_class(name, (base,), {}, lambda ns: ns.update(attrs))
        logging.debug(f"\t\tDeclare Edge class `{t.__name__}` (prop: `{properties}`).")
        setattr(self.module, t.__name__, t)
        return t

    def make_transformer_class(self, transformer_type, node_type=None, properties=None, edge=None, columns=None, output_validator=None, **kwargs):
        """
        Create a transformer class with the given parameters.

        Args:
            transformer_type: The class of the transformer.
            node_type: The type of the target node created by the transformer.
            properties: The properties of the transformer.
            edge: The edge type of the transformer.
            columns: The columns to be processed by the transformer.
            output_validator: validate.OutputValidator instance for transformer output validation.
            **kwargs: Additional keyword arguments.

        Returns:
            The created transformer class.

        Raises:
            TypeError: If the transformer type is not an existing transformer.
        """
        if hasattr(transformer, transformer_type):
            parent_t = getattr(transformer, transformer_type)
            kwargs.setdefault("subclass", parent_t)
            if not issubclass(parent_t, base.Transformer):
                self.error(f"Object `{transformer_type}` is not an existing transformer.", exception = exceptions.DeclarationError)
            else:
                if node_type:
                    nt = node_type.__name__
                else:
                    nt = "."
                logging.debug(f"\t\tDeclare Transformer class '{transformer_type}' for node type '{nt}'")
                return parent_t(target=node_type, properties_of=properties, edge=edge, columns=columns, output_validator=output_validator, **kwargs)
        else:
            # logging.debug(dir(generators))
            self.error(f"Cannot find a transformer class with name `{transformer_type}`.", exception = exceptions.DeclarationError)

class YamlParser(Declare):
    """
    Parse a table extraction configuration and return the three objects needed to configure an Adapter.

    The config is a dictionary containing only strings, as converted from the following YAML description:

    .. code-block:: yaml

            row:
               map:
                  columns:
                    - <MY_COLUMN_NAME>
                  to_subject: <MY_SUBJECT_TYPE>
            transformers:
                - map:
                    columns:
                        - <MY_COLUMN_NAME>
                    to_object: <MY_OBJECT_TYPE>
                    via_relation: <MY_RELATION_TYPE>
                - map:
                    columns:
                        - <MY_OTHER_COLUMN>
                    to_property:
                        - <MY_PROPERTY>
                    for_objects:
                        - <MY_OBJECT_TYPE>

    This maps the table row to a MY_SUBJECT_TYPE node type, adding an edge of type MY_RELATION_TYPE,
    between the MY_SUBJECT_TYPE node and another MY_OBJECT_TYPE node. The data in MY_OTHER_COLUMN is mapped
    to the MY_PROPERTY property of the MY_OBJECT_TYPE node. Note that `to_properties` may effectively map to
    an edge type or several types.

    In order to allow the user to write mappings configurations using their preferred vocabulary, the following
    keywords are interchangeable:
        - subject = row = entry = line,
        - columns = fields,
        - to_target = to_object = to_node
        - via_edge = via_relation = via_predicate.

    :param dict config: A configuration dictionary.
    :param module: The module in which to insert the types declared by the configuration.
    :return tuple: subject_transformer, transformers, metadata as needed by the Adapter.
    """

    def __init__(self, config: dict, module=types):
        """
        Initialize the YamlParser.

        Args:
            config (dict): The configuration dictionary.
            module: The module in which to insert the types declared by the configuration.
        """
        super().__init__(module)
        self.config = config

        logging.debug(f"Classes will be created in module '{self.module}'")

    def get_not(self, keys, pconfig=None):
        """
        Get the first dictionary (key, item) not matching any of the passed keys.

        Args:
            keys: The keys to exclude.
            pconfig: The configuration dictionary to search in (default is self.config).

        Returns:
            dict: The first dictionary not matching any of the passed keys.
        """
        res = {}
        if not pconfig:
            pconfig = self.config
        for k in pconfig:
            if k not in keys:
                res[k] = pconfig[k]
        return res

    def get(self, keys, pconfig=None):
        """
        Get a dictionary item matching any of the passed keys.

        Args:
            keys: The keys to search for.
            pconfig: The configuration dictionary to search in (default is self.config).

        Returns:
            The first item matching any of the passed keys, or None if no match is found.
        """
        if not pconfig:
            pconfig = self.config
        for k in keys:
            if k in pconfig:
                return pconfig[k]
        return None

    def _extract_metadata(self, k_metadata_column, metadata_list, metadata, type, columns):
        """
        Extract metadata and update the metadata dictionary.

        Args:
            k_metadata_column (list): List of keys to be used for adding source column names.
            metadata_list (list): List of metadata items to be added.
            metadata (dict): The metadata dictionary to be updated.
            type (str): The type of the node or edge.
            columns (list): List of columns to be added to the metadata.

        Returns:
            dict: The updated metadata dictionary.
        """
        if metadata_list and type:
                metadata.setdefault(type, {})
                for item in metadata_list:
                    metadata[type].update(item)
                for key in k_metadata_column:
                    if key in metadata[type]:
                        # Use the value of k_metadata_column as the key.
                        key_name = metadata[type][key]
                        # Remove the k_metadata_column key from the metadata dictionary.
                        if key_name in metadata[type]:
                            msg = f"They key you used for adding source column names: `{key_name}` to node: `{type}` already exists in the metadata dictionary."
                            logging.error(msg)
                            raise KeyError(msg)
                        del metadata[type][key]
                        if columns:
                            metadata[type][key_name] = ", ".join(columns)

                return metadata
        else:
            return None

    def __call__(self):
        """
        Parse the configuration and return the subject transformer and transformers.

        Returns:
            tuple: The subject transformer and a list of transformers.
        """
        logging.debug(f"Parse mapping:")

        properties_of = {}
        transformers = []
        metadata = {}

        # Various keys are allowed in the config to allow the user to use their favorite ontology vocabulary.
        k_row = ["row", "entry", "line", "subject", "source"]
        k_subject_type = ["to_subject"]
        k_columns = ["columns", "fields", "column"]
        k_target = ["to_target", "to_object", "to_node"]
        k_subject = ["from_subject", "from_source"]
        k_edge = ["via_edge", "via_relation", "via_predicate"]
        k_properties = ["to_properties", "to_property"]
        k_prop_to_object = ["for_objects", "for_object"]
        k_transformer = ["transformers"]
        k_metadata = ["metadata"]
        k_metadata_column = ["add_source_column_names_as"]
        k_validate = ["validate"]
        k_validate_output = ["validate_output"]

        transformers_list = self.get(k_transformer)

        # First, parse subject's type
        logging.debug(f"Declare subject type...")
        subject_dict = self.get(k_row)
        subject_transformer_class = list(subject_dict.keys())[0]
        subject_type = self.get(k_subject_type, subject_dict[subject_transformer_class])
        subject_kwargs = self.get_not(k_subject_type + k_columns, subject_dict[subject_transformer_class])
        subject_columns = self.get(k_columns, subject_dict[subject_transformer_class])
        if subject_columns != None and type(subject_columns) != list:
            logging.debug(f"\tDeclared singular subject’s column `{subject_columns}`")
            assert(type(subject_columns) == str)
            subject_columns = [subject_columns]
        logging.debug(f"\tDeclare subject of type: '{subject_type}', subject transformer: '{subject_transformer_class}', "
                      f"subject kwargs: '{subject_kwargs}', subject columns: '{subject_columns}'")

        # Parse the validation rules for the output of the subject transformer.
        s_output_validation_rules = self.get(k_validate_output, subject_dict[subject_transformer_class])
        s_yaml_output_validation_rules = yaml.dump(s_output_validation_rules, default_flow_style=False)
        s_output_validator = validate.OutputValidator()
        s_output_validator.update_rules(pa.DataFrameSchema.from_yaml(s_yaml_output_validation_rules))

        # Then, parse property mappings.
        logging.debug(f"Parse properties...")
        for n_transformer,transformer_types in enumerate(transformers_list):
            for transformer_type, field_dict in transformer_types.items():
                if not field_dict:
                    self.error(f"There is no field for the {n_transformer}th transformer: '{transformer_type}', did you forget an indentation?", "transformers", n_transformer, exception = exceptions.MissingFieldError)

                if any(field in field_dict for field in k_properties):
                    object_types = self.get(k_prop_to_object, pconfig=field_dict)
                    property_names = self.get(k_properties, pconfig=field_dict)
                    if type(property_names) != list:
                        logging.debug(f"\tDeclared singular property")
                        assert(type(property_names) == str)
                        property_names = [property_names]
                    if not object_types:
                        logging.info(f"No `for_objects` defined for properties {property_names}, I will attach those properties to the row subject `{subject_type}`")
                        object_types = [subject_type]
                    if type(object_types) != list:
                        logging.debug(f"\tDeclared singular for_object: `{object_types}`")
                        assert(type(object_types) == str)
                        object_types = [object_types]

                    column_names = self.get(k_columns, pconfig=field_dict)
                    if  column_names != None and type(column_names) != list:
                        logging.debug(f"\tDeclared singular column `{column_names}`")
                        assert(type(column_names) == str)
                        column_names = [column_names]
                    gen_data = self.get_not(k_target + k_edge + k_columns, pconfig=field_dict)
                    prop_transformer = self.make_transformer_class(transformer_type, columns=column_names, **gen_data)

                    for object_type in object_types:
                        properties_of.setdefault(object_type, {})
                        for property_name in property_names:
                            properties_of[object_type].setdefault(prop_transformer, property_name)
                        logging.debug(f"\t\tDeclared property mapping for `{object_type}`: {properties_of[object_type]}")


        metadata_list = self.get(k_metadata)

        logging.debug(f"Parse subject transformer...")
        source_t = self.make_node_class(subject_type, properties_of.get(subject_type, {}))
        subject_transformer = self.make_transformer_class(
            columns=subject_columns, transformer_type=subject_transformer_class,
            node_type=source_t, properties=properties_of.get(subject_type, {}), output_validator=s_output_validator, **subject_kwargs)
        logging.debug(f"\tDeclared subject transformer: {subject_transformer}")

        extracted_metadata = self._extract_metadata(k_metadata_column, metadata_list, metadata, subject_type, subject_columns)
        if extracted_metadata:
            metadata.update(extracted_metadata)


        # Then, declare types.
        logging.debug(f"Declare types...")
        for n_transformer,transformer_types in enumerate(transformers_list):
            for transformer_type, field_dict in transformer_types.items():
                if not field_dict:
                    continue
                elif any(field in field_dict for field in k_properties):
                    if any(field in field_dict for field in k_target):
                        prop = self.get(k_properties, field_dict)
                        target = self.get(k_target, field_dict)
                        self.error(f"ERROR in transformer '{transformer_type}': one cannot "
                                      f"declare a mapping to both properties '{prop}' and object type '{target}'.", "transformers", n_transformer, exception = exceptions.CardinalityError)
                    continue
                else:
                    if type(field_dict) != dict:
                        raise Exception(str(field_dict)+" is not a dictionary")

                    columns = self.get(k_columns, pconfig=field_dict)
                    if type(columns) != list:
                        logging.debug(f"\tDeclared singular column")
                        assert(type(columns) == str)
                        columns = [columns]

                    target = self.get(k_target, pconfig=field_dict)
                    if type(target) == list:
                        self.error(f"You cannot declare multiple objects in transformers. For transformer `{transformer_type}`.", section="transformers", index=n_transformer, indent=1, exception = exceptions.CardinalityError)

                    subject = self.get(k_subject, pconfig=field_dict)
                    if type(subject) == list:
                        self.error(f"You cannot declare multiple subjects in transformers. For transformer `{transformer_type}`.", section="transformers", index=n_transformer, indent=1, exception = exceptions.CardinalityError)

                    edge = self.get(k_edge, pconfig=field_dict)
                    if type(edge) == list:
                        self.error(f"You cannot declare multiple relations in transformers. For transformer `{transformer_type}`.", section="transformers", index=n_transformer, indent=1, exception = exceptions.CardinalityError)

                    gen_data = self.get_not(k_target + k_edge + k_columns, pconfig=field_dict)

                    # Harmonize the use of the `from_subject` and `from_source` synonyms in the configuration, because
                    # from_subject` is used in the transformer class to refer to the source node type.
                    if 'from_source' in gen_data:
                        gen_data['from_subject'] = gen_data['from_source']
                        del gen_data['from_source']

                    if target and edge:
                        logging.debug(f"\tDeclare node .target for `{target}`...")
                        target_t = self.make_node_class(target, properties_of.get(target, {}))
                        logging.debug(f"\t\tDeclared target for `{target}`: {target_t.__name__}")
                        if subject:
                            logging.debug(f"\tDeclare subject for `{subject}`...")
                            subject_t = self.make_node_class(subject, properties_of.get(subject, {}))
                            edge_t = self.make_edge_class(edge, subject_t, target_t, properties_of.get(edge, {}))
                        else:
                            logging.debug(f"\tDeclare edge for `{edge}`...")
                            edge_t = self.make_edge_class(edge, source_t, target_t, properties_of.get(edge, {}))

                        # Parse the validation rules for the output of the transformer.
                        output_validation_rules = self.get(k_validate_output, pconfig=field_dict)
                        yaml_output_validation_rules = yaml.dump(output_validation_rules, default_flow_style=False)
                        output_validator = validate.OutputValidator()
                        output_validator.update_rules(pa.DataFrameSchema.from_yaml(yaml_output_validation_rules))

                        logging.debug(f"\tDeclare transformer `{transformer_type}`...")
                        transformers.append(self.make_transformer_class(
                            transformer_type=transformer_type, node_type=target_t,
                            properties=properties_of.get(target, {}), edge=edge_t, columns=columns, output_validator=output_validator, **gen_data))
                        logging.debug(f"\t\tDeclared mapping `{columns}` => `{edge_t.__name__}`")
                    elif (target and not edge) or (edge and not target):
                        self.error(f"Cannot declare the mapping  `{columns}` => `{edge}` (target: `{target}`), missing either an object or a relation.", "transformers", n_transformer, indent=2, exception = exceptions.MissingDataError)

                    extracted_metadata = self._extract_metadata(k_metadata_column, metadata_list, metadata, target, columns)
                    if extracted_metadata:
                        metadata.update(extracted_metadata)

                    if edge:
                        extracted_metadata = self._extract_metadata(k_metadata_column, metadata_list, metadata, edge, None)
                        if extracted_metadata:
                            metadata.update(extracted_metadata)

        # Extract input data validation schema from yaml file and instantiate a Pandera DataFrameSchema object and validator.
        validation_rules = self.get(k_validate)
        yaml_validation_rules = yaml.dump(validation_rules, default_flow_style=False)
        validation_schema = pa.DataFrameSchema.from_yaml(yaml_validation_rules)

        logging.debug(f"source class: {source_t}")
        logging.debug(f"properties_of: {properties_of}")
        logging.debug(f"transformers: {transformers}")
        logging.debug(f"metadata: {metadata}")
        return subject_transformer, transformers, metadata, validate.InputValidator(validation_schema)

