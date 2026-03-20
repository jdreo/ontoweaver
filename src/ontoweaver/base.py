""" Interfaces for OntoWeaver.
"""
import logging
import math

import types as pytypes
import pandas as pd

from collections.abc import Iterable
from abc import ABCMeta as ABSTRACT, abstractmethod
from abc import abstractmethod as abstract
from typing import TypeAlias
from typing import Optional
from enum import Enum, EnumMeta

from . import errormanager
from . import serialize
from . import exceptions
from . import validate
from . import types as owtypes

logger = logging.getLogger("ontoweaver")


def is_not_null(val):
    """
    Checks if cell value is not empty nor 'null', 'nan'...

    Args:
        val: The value to check.

    Returns:
        bool: True if the value is valid, False otherwise.
    """
    if pd.api.types.is_numeric_dtype(type(val)):
        if (math.isnan(val) or val == float("nan")):
            return False
    elif str(val).lower() == "nan":  # Conversion from Pandas' `object` needs to be explicit.
        return False
    elif str(val) == "":
        return False
    elif str(val) == 'None':
        return False
    return True


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


class ErrorManager:
    def __init__(self, raise_errors = True):
        self.raise_errors = raise_errors

    def error(self, msg, section = None, index = None, exception = exceptions.OntoWeaverError, indent = 0):

        location = ""
        if section:
            location = f" [for {section}"
            if index:
                location += f" #{index}"
            location += "]"

        err = "\t"*indent
        err += msg
        err += location

        if self.raise_errors:
            raise exception(err)
        else:
            logger.error(err)

        return err


class Element(metaclass = ABSTRACT):
    """Base class for either Node or Edge.

    Manages allowed properties mechanics."""

    def __init__(self,
                 id        : Optional[str] = None,
                 properties: Optional[dict[str,str]] = {},
                 label     : Optional[str] = None,
                 serializer: Optional[serialize.Serializer] = serialize.All(),
                 ):
        """Instantiate an element.

        :param str id: Unique identifier of the element. If id == None, is then set to the empty string.
        :param dict[str,str] properties: All available properties for this instance.
        :param str label: The label of the element.
        :param serializer: an instance indicating what to serialize to get a unique ID
        """
        if not id:
            self._id = ''
        else:
            self._id = str(id)

        # Use the setter to get sanity checks.
        self.properties = properties

        if not label:
            # Do not change the name here, or BioCypher will have problem
            # finding back labels.
            self._label = self.__class__.__name__ 
        else:
            self._label = str(label)

        self.serializer = serializer

    def __str__(self):
        return self.serializer(self)

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    @staticmethod
    @abstract
    def fields() -> list[str]:
        """List of property fields provided by the (sub)class."""
        raise NotImplementedError

    @abstract
    def as_tuple(self):
        """Convert the element class into Biocypher's expected tuple.

        Filter out properties along the way.
        """
        raise NotImplementedError

    @classmethod
    @abstract
    def from_tuple(cls,
                   biocypher_tuple : tuple,
                   serializer: Optional[serialize.Serializer] = serialize.All()
        ):
        # return cls(biocypher_tuple,serializer)
        raise NotImplementedError

    @property
    def id(self) -> str:
        return self._id

    @property
    def label(self) -> str:
        return self._label

    @property
    def properties(self) -> dict[str,str]:
        return self._properties

    @properties.setter
    def properties(self, properties: dict[str,str]):
        """Set available properties.

        Asserts that the passed properties are in the declared fields."""

        # Sanity checks:
        assert(properties is not None)
        assert(type(properties) == dict)
        # logger.debug(f"Properties of `{type(self).__name__}`: {list(properties.keys())}, available: {list(self.available())}")
        # TODO enable the usage of available() function to disable / enable parts of ontology / certain nodes
        # for p in properties:
        #     if p not in self.available():
        #         logger.error(f"\t\tProperty `{p}` should be available for type `{type(self).__name__}`, available ones: `{list(self.available())}`")
        #         assert(p in self.available())
        self._properties = properties


class Node(Element):
    """Base class for any Node."""

    def __init__(self,
                 id        : Optional[str] = None,
                 properties: Optional[dict[str,str]] = {},
                 label     : Optional[str] = None,  # Set from subclass name.
                 serializer: Optional[serialize.Serializer] = serialize.node.All(),
                 ):
        """Instantiate a Node.

        :param str id: Unique identifier of the node. If id == None, is then set to the empty string.
        :param dict[str,str] properties: All available properties for this instance.
        :param str label: The label of the node.
        :param serializer: an instance indicating what to serialize to get a unique ID
        """
        super().__init__(id = id, properties = properties, label = label, serializer = serializer)

    Tuple: TypeAlias = tuple[str,str,dict[str,str]]
    def as_tuple(self) -> Tuple:
        """Export the Node as a Biocypher tuple."""
        return (
            self._id,
            self._label,
            # FIXME this has been changed to keep ALL properties. No checking if allowed
            self.properties
        )

    @classmethod
    def from_tuple(cls,
                   biocypher_tuple : tuple[str,str,dict[str,str]],
                   serializer: Optional[serialize.Serializer] = serialize.node.All(),
                   ):
        assert(len(biocypher_tuple) == 3)
        return cls(
            id         = biocypher_tuple[0],
            label      = biocypher_tuple[1],
            properties = biocypher_tuple[2],
            serializer = serializer)

    def fields(self) -> list[str]:
        """List of property fields provided by the (sub)class."""
        return list(self.properties.keys())

    def __str__(self):
        return self.serializer(self)

    def __repr__(self):
        return f"({self.label}:{self.id}/{self.properties})"

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    # def serialize(self):
    #     return {
    #         "id": self._id,
    #         "label": self._label,
    #         "properties": self.properties,
    #         "serializer": self.serializer
    #     }

class Edge(Element):
    """Base class for any Edge."""

    def __init__(self,
                 id        : Optional[str] = None,
                 id_source : Optional[str] = None,
                 id_target : Optional[str] = None,
                 properties: Optional[dict[str,str]] = {},
                 label     : Optional[str] = None,  # Set from subclass name.
                 serializer: Optional[serialize.Serializer] = serialize.edge.All(),
                 ):
        """Instantiate an Edge.

        :param str id: Unique identifier of the edge. If id == None, is then set to the empty string.
        :param str id_source: Unique identifier of the source Node. If None, is then set to the empty string.
        :param str id_target: Unique identifier of the target Node. If None, is then set to the empty string.
        :param dict[str,str] properties: All available properties for this instance.
        :param str label: The label of the node.
        :param comparer: an instance indicating what to serialize to get a unique ID
        """
        super().__init__(id = id, properties = properties, label = label, serializer = serializer)
        self._id_source = str(id_source)
        self._id_target = str(id_target)

    @staticmethod
    @abstract
    def source_type():
        raise NotImplementedError

    @staticmethod
    @abstract
    def target_type():
        raise NotImplementedError

    @property
    def id_source(self):
        return self._id_source

    @id_source.setter
    def id_source(self, id_source):
        self._id_source = id_source

    @property
    def id_target(self):
        return self._id_target

    @id_target.setter
    def id_target(self, id_target):
        self._id_target = id_target

    Tuple: TypeAlias = tuple[str,str,str,dict[str,str]]
    def as_tuple(self) -> Tuple:
        """Export the Edge as a Biocypher tuple."""
        return (
            self._id,
            self._id_source,
            self._id_target,
            self._label,
            #.FIXME no checking if properties are allowed allowed_properties()
            self.properties
        )

    @classmethod
    def from_tuple(cls,
                   biocypher_tuple : tuple[str,str,str,str,dict[str,str]],
                   serializer: Optional[serialize.Serializer] = serialize.edge.All()
                   ):
        assert(len(biocypher_tuple) == 5)
        logging.debug(biocypher_tuple)
        return cls(
            id         = biocypher_tuple[0],
            id_source  = biocypher_tuple[1],
            id_target  = biocypher_tuple[2],
            properties = biocypher_tuple[4],
            label      = biocypher_tuple[3],
            serializer = serializer)

    def __repr__(self):
        if self.source_type() == Node:
            st = "."
        elif self.source_type() == None:
            st = ""
        else:
            st = f"{self.source_type().__name__}"
        if self.target_type() == Node:
            tt = "."
        else:
            tt = f"{'>'.join([t.__name__ for t in self.target_type()])}"

        return f"<({st}:{self.id_source})--[{self.label}:{self.id}/{self.properties}]-->({tt}:{self.id_target})>"

    def fields(self) -> list[str]:
        """List of property fields provided by the (sub)class."""
        return list(self.properties.keys())

    def __str__(self):
        return self.serializer(self)

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    # def serialize(self):
    #     return {
    #         "id": self._id,
    #         "id_source": self._id_source,
    #         "id_target": self._id_target,
    #         "label": self._label,
    #         "properties": self.properties,
    #         "serializer": self.serializer
    #     }

class GenericEdge(Edge):
    """Base class for any Edge."""

    def __init__(self,
                 id        : Optional[str] = None,
                 id_source : Optional[str] = None,
                 id_target : Optional[str] = None,
                 properties: Optional[dict[str,str]] = {},
                 label     : Optional[str] = None,  # Set from subclass name.
                 serializer: Optional[serialize.Serializer] = serialize.edge.All(),
                 ):
        """Instantiate an Edge.

        :param str id: Unique identifier of the edge. If id == None, is then set to the empty string.
        :param str id_source: Unique identifier of the source Node. If None, is then set to the empty string.
        :param str id_target: Unique identifier of the target Node. If None, is then set to the empty string.
        :param dict[str,str] properties: All available properties for this instance.
        :param str label: The label of the node.
        :param comparer: an instance indicating what to serialize to get a unique ID
        """
        super().__init__(id = id, id_source = id_source, id_target = id_target, properties = properties, label = label, serializer = serializer)

        logging.debug(f"GenericEdge ID: {id}")

    @staticmethod
    def source_type():
        return Node

    @staticmethod
    def target_type():
        return Node



class Adapter(errormanager.ErrorManager, metaclass = ABSTRACT):
    """Base class for implementing an adapter that consumes tabular data."""

    def __init__(self, raise_errors = True):
        """Allow to indicate which Element subclasses and which property fields
        are allowed to be exported by Biocypher.
        """
        self._nodes = []
        self._edges = []
        self.errors = []
        super().__init__(raise_errors)


    def nodes_append(self, node_s) -> None:
        """Append an Node (or each Node in a list of nodes) to the internal list of nodes."""
        if issubclass(type(node_s), Node):
            nodes = [node_s]
        else:
            nodes = node_s

        # logger.debug(f"Nodes: {nodes}.")
        for node in nodes:
            # logger.debug(f"\tAppend node {node}.")
            # Checking for duplicates in reconciliation, otherwise complexity too high.
            self._nodes.append(node.as_tuple())
            # return True

    def edges_append(self, edge_s) -> None:
        """Append an Edge (or each Edge in a list of edges) to the internal list of edges."""
        if issubclass(type(edge_s), Edge):
            edges = [edge_s]
        else:
            edges = edge_s

        # logger.debug(f"Edges: {edges}.")
        for edge in edges:
            # logger.debug(f"\tAppend edge {edge}.")
            # Checking for duplicates in reconciliation, otherwise complexity too high.
            self._edges.append(edge.as_tuple())
            # return True

    @property
    def nodes(self) -> Iterable[Node.Tuple]:
        """Return a generator yielding nodes."""
        for n in self._nodes:
            yield n

    @property
    def edges(self) -> Iterable[Edge.Tuple]:
        """Return a generator yielding edges."""
        for e in self._edges:
            yield e

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    def __call__(self):
        for local_nodes, local_edges in self.run():
            yield local_nodes, local_edges


class Declare(errormanager.ErrorManager):
    """
    Declarations of functions used to declare and instantiate object classes used by the Adapter for the mapping
    of the data frame.

    Args:
        module: The module in which to insert the types declared by the configuration.
    """

    def __init__(self,
                 module=owtypes,
                 raise_errors = True,
                 ):
        super().__init__(raise_errors)
        self.module = module

        self.declared = []


    def make_node_class(self, name, properties={}, base=Node):
        """
        LabelMaker a node class with the given name and properties.

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
            logger.debug(
                f"\t\tNode class `{name}` (prop: `{cls.fields()}`) already exists, I will not create another one.")
            for p in properties.values():
                if p not in cls.fields():
                    logger.warning(f"\t\t\tProperty `{p}` not found in declared fields for node class `{cls.__name__}`.")
            return cls

        def fields():
            return list(properties.values())

        attrs = {
            "__module__": self.module.__name__,
            "fields": staticmethod(fields),
        }
        t = pytypes.new_class(name, (base,), {}, lambda ns: ns.update(attrs))
        logger.debug(f"\t\tDeclare Node class `{t.__name__}` (prop: `{properties}`).")
        setattr(self.module, t.__name__, t)

        self.declared.append(t)
        return t


    def get_node_class(self, name):
        return getattr(self.module, name)


    def make_edge_class(self, name, source_t, target_t, properties={}, base=Edge):
        """
        LabelMaker an edge class with the given name, source type, target type, and properties.

        Args:
            name: The name of the edge class.
            source_t: The source type of the edge.
            target_t: The target type of the edge.
            properties (dict): The properties of the edge class.
            base: The base class for the edge class.

        Returns:
            The created edge class.
        """
        # If type already exists, check if the fields are the same.
        if hasattr(self.module, name):
            cls = getattr(self.module, name)
            cls_fields = cls.fields()

            # Compare the properties with the existing class fields
            if properties == cls_fields:
                logger.debug(
                    f"\t\tEdge class `{name}` already exists with the same properties,"
                     " I will not create another one."
                 )
                logger.debug(f"\t\t\tProperties: `{cls_fields}`")
                return cls

            logger.warning(f"\t\tEdge class `{name}` already exists, but properties do not match.")
            # If properties do not match, we proceed to create a new class with the new properties.
            # FIXME: Would make much more sense to just append(?) new properties to existing class instead of creating new class.

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
        logger.debug(f"\t\tDeclare Edge class `{t.__name__}` (prop: `{properties}`).")
        setattr(self.module, t.__name__, t)

        self.declared.append(t)
        return t


    def get_edge_class(self, name):
        return getattr(self.module, name)


class MappingParser(Declare):
    # Various keys are allowed in the config to allow the user to use their favorite ontology vocabulary.
    k_row = ["row", "entry", "line", "subject", "source"]
    k_subject_type = ["to_subject", "to_object', 'to_node", "to_label", "to_type", "id_from_column", "id_from_element"]
    k_columns = ["columns", "fields", "column", "field", "element", "match_column", "id_from_column", "match_element", "id_from_element", "key", "keys"]
    k_target = ["to_target", "to_object", "to_node", "to_label", "to_type"]
    k_subject = ["from_subject", "from_source", "to_subject", "to_source", "to_node", "to_label", "to_type"]
    k_edge = ["via_edge", "via_relation", "via_predicate"]
    k_properties = ["to_properties", "to_property"]
    k_prop_to_object = ["for_objects", "for_object", "for_relation", "for_relations", "for_edge", "for_edges"]
    k_transformer = ["transformers"]
    k_metadata = ["metadata"]
    k_metadata_column = ["add_source_column_names_as"]
    k_validate_output = ["validate_output"]
    k_final_type = ["final_type", "final_object", "final_node", "final_subject", "final_label", "final_target"]
    k_reverse_edge = ["reverse_relation", "reverse_edge", "reverse_predicate", "reverse_link"]
    k_match_type_from = ["match_type_from_column", "match_type_from_element"]


class Transformer(errormanager.ErrorManager):
    """"Class used to manipulate cell values and return them in the correct format."""""

    def __init__(self,
        properties_of,
        value_maker = None,
        label_maker = None,
        branching_properties = None,
        columns = None,
        output_validator = None,
        multi_type_dict = None,
        raise_errors = True,
        **kwargs
    ):
        """
        Make a transformer class with the given parameters.
        Instantiate transformers.

        :param properties_of: the properties of each node type.
        :param value_maker: the ValueMaker object used for the logic of cell value selection for each transformer. Default is None.
        :param label_maker: the LabelMaker object used for handling the creation of the output of the transformer. Default is None.
        :param branching_properties: in case of branching on cell values, the dictionary holds the properties for each branch.
        :param columns: the columns to use in the mapping.
        :param output_validator: the OutputValidator object used for validating transformer output. Default is None.
        :param multi_type_dict: the dictionary holding regex patterns for node and edge type branching based on cell values.
        :param raise_errors: whether to raise errors or not. Default is True.
            each transformer is instantiated with a default OutputValidator object, and additional user-defined rules if needed in
            the tabular module.


        """
        super().__init__(raise_errors)

        self.properties_of = properties_of
        self.value_maker = value_maker
        self.label_maker = label_maker
        self.branching_properties = branching_properties
        self.columns = columns
        self.output_validator = output_validator
        if not self.output_validator:
            self.output_validator = validate.OutputValidator(validate.default_validation_rules, raise_errors = raise_errors)
        self.parameters = kwargs
        self.multi_type_dict = multi_type_dict
        self.final_type = None # The final type is to be passed by the label maker class based on the YAML mapping. That
                               # is why here it is None by default
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.declare_types = Declare()



    def get_transformer(self):
        return self


    def __call__(self, row, i):
        """
        Process a row. If not empty/null, yields concatenated items as node IDs.

        Args:
            row: The current row of the DataFrame.
            i: The index of the current row.

        Yields:
            str: The concatenated string from the cell values.
        """
        for val in self.value_maker(self.columns, row, i):
            value, edge_type, node_type, reverse_edge = self.create(val, row)
            if is_not_null(value):
                yield value, edge_type, node_type, reverse_edge


    #FIXME: The functions below are never implemented.
    @abstractmethod
    def nodes(self):
        raise NotImplementedError

    @abstractmethod
    def edges(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def edge_type():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def target_type():
       raise NotImplementedError

    @classmethod
    def source_type(cls):
       return cls.edge_type().source_type()

    def __repr__(self):

        representation = ""

        if hasattr(self, "from_subject"):
            from_subject = self.from_subject
        else:
            from_subject = "."

        target_name = ""
        edge_name = ""

        if hasattr(self, "multi_type_dict") and self.multi_type_dict is not None:
            for key, value in self.multi_type_dict.items():
                if value['to_object'] and value['via_relation']:
                    target_name = value['to_object']
                    edge_name = value['via_relation']
                elif value['to_object'] and not value['via_relation']:
                    target_name = value['to_object']
                    edge_name = "."


                if self.properties_of:
                    # The transformer is not a branching transformer, and has only one set of properties.
                    props = "{"
                    trans = []
                    for kind in self.properties_of:
                        trans.append(f"{kind} {self.properties_of[kind]}>")
                    props += ", ".join(trans)
                    props+="}"
                elif self.branching_properties and self.branching_properties.get(value['to_object'], None):
                    # The transformer is a branching transformer, and has multiple sets of properties. We extract the ones for the current type.
                    props = self.branching_properties.get(value['to_object'])
                else:
                    # The transformer has no properties for the type.
                    props = "{}"


                params = ""
                parameters = {k:v for k,v in self.parameters.items() if k not in ['subclass', 'from_subject', "match"]}
                if parameters:
                    p = []
                    for k,v in parameters.items():
                        p.append(f"{k}={v}")
                    params = ','.join(p)

                if from_subject == "." and edge_name == "." and target_name == "." and props == "{}":
                    # If this is a property transformer
                    link = ""

                elif from_subject == "." and edge_name == "." and (target_name != "." or props != "{}"):
                    # This a subject transformer.
                    link = f" => ({target_name}/{props})"

                else:
                    # This is a regular transformer.
                    link = f" => ({from_subject})--[{edge_name.__name__}]->({target_name.__name__}/{props})"

                if self.columns:
                    columns = self.columns
                else:
                    columns = []

                # for c in columns:
                    # if not isinstance(c, str):
                    #     self.error(f"Column `{c}` is not a string, did you mistype a leading colon?", exception=exceptions.ParsingError)
                columns = [str(i) for i in columns]

                representation += (f"<{type(self).__name__}({params}) {','.join(columns)}{link}>")

        else:
            #The transformer is a property transformer. We add the property name and value in the tabular.properties() function.
            representation += (f"<{type(self).__name__}() {','.join(self.columns) if self.columns else ''} =>")

        return representation


    def validate(self, res):
        """
        Validate the output of the transformer, using the output_validator. of the transformer instance.
        """
        try:
            if isinstance(self.output_validator, validate.SimpleOutputValidator) or isinstance(self.output_validator, validate.SkipValidator):
                # The SimpleOutputValidator and SkipValidator do not use the Pandera package, and the value is therefore not
                # required to be in a DataFrame format. Voiding the creation of a DataFrame here saves computational time for
                # large datasets.
                if self.output_validator(res):
                    return True
                else:
                    return False
            else:
                if self.output_validator(pd.DataFrame([res], columns=["cell_value"])):
                    return True
                else:
                    return False
        except pa.errors.SchemaErrors as error:
            msg = f"Transformer {self.__repr__()} did not produce valid data {error}."
            self.error(msg, exception = exceptions.DataValidationError)


    def create(self, returned_value, row):
        """
        Create the output of the transformer, using the label_maker of the transformer instance.

        Returns:
            - Extracted cell value (can be node ID, property value, edge ID),
            - edge type,
            - target node type, and
            - reverse relation in case declared in the mapping.
        """
        result_object = self.label_maker(self.validate, returned_value, self.multi_type_dict, self.branching_properties, row)
        if not result_object:
            return None, None, None, None
        if result_object.target_node_type:
            self.target_type = result_object.target_node_type.__name__
        if result_object.target_element_properties is not None:
            self.properties_of = result_object.target_element_properties
        self.final_type = result_object.final_type
        return result_object.extracted_cell_value, result_object.edge_type, result_object.target_node_type, result_object.reverse_relation


class All:
    """Gathers lists of subclasses of Element and their fields
    existing in a given module.

    Is generally used to label_maker an `all` variable in a module:
    .. code-block:: python

        all = base.All(sys.modules[__name__])

    Which can later be used to pass all available Element types to an Adapter:
    .. code-block:: python

        a = MyAdapter( node_types = MyModule.all.nodes() )
    """

    def __init__(self, module):
        self.module = module

    def elements(self, asked: Element = Element) -> list[Element]:
        m = self.module.__dict__
        classes = []
        for c in m:
            if isinstance(m[c], type) \
            and m[c].__module__ == self.module.__name__ \
            and issubclass(m[c], asked):
                classes.append(m[c])
                logger.debug(f"Found `{asked.__name__}` class: `{m[c]}` (prop: `{m[c].fields()}`).")
                # t = m[c]
                # logger.debug(f"\t\t#### {t.mro()[:-3]}/{t.__name__} => {t.fields()}")
        return classes

    def nodes(self) -> list[Node]:
        return self.elements(Node)

    def edges(self) -> list[Edge]:
        return self.elements(Edge)

    def node_fields(self) -> list[str]:
        names = [] # FIXME use a set?
        for c in self.nodes():
            names += c.fields()
        return names

    def edge_fields(self) -> list[str]:
        names = [] # FIXME use a set?
        for c in self.edges():
            names += c.fields()
        return names

