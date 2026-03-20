""" Adapters managing iterable data.
"""

import sys
import logging
import threading

import pandas as pd

from typing import Optional
from itertools import chain
from alive_progress import alive_bar
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from abc import ABCMeta as ABSTRACT, abstractmethod

from . import base
from . import transformer
from . import exceptions
from . import validate

logger = logging.getLogger("ontoweaver")

class IterativeAdapter(base.Adapter, metaclass = ABSTRACT):
    """Base class for implementing a Biocypher adapter that consumes iterative data."""

    def __init__(self,
                 subject_transformer: base.Transformer,
                 transformers: Iterable[base.Transformer],
                 metadata: Optional[dict] = None,
                 validator: Optional[validate.InputValidator] = None,
                 type_affix: Optional[base.TypeAffixes] = base.TypeAffixes.suffix,
                 type_affix_sep: Optional[str] = ":",
                 parallel_mapping: int = 0,
                 raise_errors = True,
                 progress_bar = False,
                 ):
        """
        Instantiate the adapter.

        Args:
            subject_transformer (base.Transformer): The transformer that maps the subject node.
            transformers (Iterable[base.Transformer]): List of transformer instances that map the data frame to nodes and edges.
            metadata (Optional[dict]): Metadata to be added to all the nodes and edges.
            type_affix (Optional[TypeAffixes]): Where to add a type annotation to the labels (either TypeAffixes.prefix, TypeAffixes.suffix or TypeAffixes.none).
            type_affix_sep (Optional[str]): String used to separate a label from the type annotation (WARNING: double-check that your BioCypher config does not use the same character as a separator).
            parallel_mapping (int): Number of workers to use in parallel mapping. Defaults to 0 for sequential processing.
            raise_errors (bool): if True, will raise an exception when an error is encountered, else, will log the error and try to proceed.
            progress_bar: if True, will show a progress bar while processing the data
        """
        super().__init__(raise_errors)

        self.validator = validator

        if type_affix not in base.TypeAffixes:
            self.error(f"`type_affix`={type_affix} is not one of the allowed values ({list(base.TypeAffixes)})", exception = exceptions.ConfigError)
        else:
            self.type_affix = type_affix

        self.type_affix_sep = type_affix_sep

        self.subject_transformer = subject_transformer
        assert self.subject_transformer
        self.transformers = transformers
        self.property_transformers = [] # populated at parsing in self.properties.
        self.metadata = metadata
        # logger.debug(self.target_element_properties)
        self.parallel_mapping = parallel_mapping

        self.non_viable_rows = set()

        self.progress_bar = progress_bar


    # FIXME not used, maybe will come in handy?
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
        LabelMaker a unique ID for the given cell consisting of the entry name and type,
        taking into account affix and separator configuration.

        Args:
            entry_type: The type of the entry.
            entry_name: The name of the entry.

        Returns:
            str: The created ID.

        Raises:
            ValueError: If the ID creation fails.
        """
        assert isinstance(entry_type, str), "Entry type is not a string"
        if not isinstance(entry_name, str):
            logger.warning(f"Identifier `{entry_name}` (of type `{entry_type}`) is not a string, I had to convert it explicitely, check that the related transformer yields a string.")
            entry_name = str(entry_name)

        if '[' in entry_name or ']' in entry_name:
            logger.warning(f"Identifier `{entry_name}` (of type `{entry_type}`) contains brackets. Maybe you should use a `split` transformer for this column?")

        if self.type_affix == base.TypeAffixes.prefix:
            idt = f'{entry_type}{self.type_affix_sep}{entry_name}'
        elif self.type_affix == base.TypeAffixes.suffix:
            idt = f'{entry_name}{self.type_affix_sep}{entry_type}'
        elif self.type_affix == base.TypeAffixes.none:
            idt = f'{entry_name}'

        if idt:
            logger.debug(f"\t\tFormatted ID `{idt}` for cell value `{entry_name}` of type: `{entry_type}`")
            return idt
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


    def properties(self, property_dict, row, i, edge_t, node_t, node = False):

        """
        Extract properties of each property category for the given node type.
        If no properties are found, return an empty dictionary.

        Args:
            property_dict: Dictionary of transformer => property name
            row: The current row of the DataFrame.
            i: The index of the current row.
            edge_t: The type of the edge of the current transformer.
            node_t: The type of the node of the current transformer.
            node: True if the object created is a node, False otherwise.

        Returns:
            dict: Extracted properties.
        """
        properties = {}

        if property_dict:

            for prop_transformer, property_name in property_dict.items():
                properties[property_name] = []

                for property_value, none_node, none_edge, none_reverse_relation \
                    in prop_transformer(row, i):

                    if property_value:
                        if isinstance(property_value, list):
                            properties[property_name] += str(property_value)
                        else:
                            properties[property_name].append(str(property_value))
                        # properties[property_name] = str(property_value).replace("'", "`") # FIXME double-check why the old code needed this.
                        logger.debug(
                            f"                 {prop_transformer}" \
                            f" to property `{property_name}` with value" \
                            f" `{properties[property_name]}`." )
                    else:
                        self.error(
                            f"Failed to extract valid property with" \
                            f" {prop_transformer.__repr__()} for {i}th row.",
                            indent=2,
                            exception = exceptions.TransformerDataError )

        # Collapse any list with less than one item to a string.
        # to_delete = []
        for k,v in properties.items():
            # if len(v) == 0:
            #     to_delete.append(k)
            # el
            if len(v) == 1:
                properties[k] = v[0]

        # We don't delete empty properties, because BioCypher would complain.
        # for k in to_delete:
            # del properties[k]

        # If the metadata dictionary is not empty, add the metadata to the property dictionary.
        if self.metadata:
            if node:
                elem = node_t
            else:
                elem = edge_t
            if elem.__name__ in self.metadata:
                for key, value in self.metadata[elem.__name__].items():
                    properties[key] = value

        return properties


    def make_node(self, node_t, id, properties):
        """
        LabelMaker nodes of a certain type.

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
        LabelMaker edges of a certain type.

        Args:
            edge_t: The type of the edge.
            id_target: The ID of the target node.
            id_source: The ID of the source node.
            properties: The properties of the edge.

        Returns:
            The created edge.
        """
        default_id = f"({id_source})--[{edge_t.__name__}]->({id_target})"
        return edge_t(id = default_id, id_source=id_source, id_target=id_target, properties=properties)

    # ==========================
    # Helper functions for run
    # ==========================

    def _make_default_source_node_id(self, row, i, local_nodes, local_errors):
        """
        Helper function to create the default source node id for each row.
        Referred to as default because of possibility
        of it changing type with `from_subject` attribute in transformers.
        """

        logger.debug("\tMake subject node...")
        for source_id, subject_edge_type, subject_node_type, _ in self.subject_transformer(row, i):
        # subject_generator_list = list(self.subject_transformer(row, i))
        # logger.debug(f"\t\tSubject items: {subject_generator_list}")
        # if len(subject_generator_list) > 1:
        #     local_errors.append(self.error(
        #         f"You cannot use transformer yielding multiple IDs for the subject. "
        #         f"Subject Transformer `{self.subject_transformer}` produced multiple IDs: "
        #         f"{subject_generator_list}, I'll take the first one and pretend this"
        #         f" did not happened.",
        #         indent=2, exception=exceptions.TransformerInterfaceError))

        # elif len(subject_generator_list) == 0:
        #     logger.debug("The subject transformer did not produce any valid ID,"
        #         " I'll try to skip this entry.")
        #     return None
        #
        # source_id, subject_edge_type, subject_node_type, _ = subject_generator_list[0]

            if self.subject_transformer.final_type:
                # If a final_type attribute is present in the transformer, use it as the source node type, instead
                # of the default type.
                subject_node_type = self.subject_transformer.final_type

            assert subject_node_type

            if source_id:
                source_node_id = self.make_id(subject_node_type.__name__, source_id)

                if source_node_id:
                    logger.debug(f"\t\tDeclared subject ID: {source_node_id}")
                    local_nodes.append(
                        self.make_node(
                            node_t = subject_node_type,
                            id = source_node_id,
                            # FIXME: Should we use the meta-way of accessing node properties as well?
                            # FIXME: This would require a refactoring of the transformer interfaces and tabular.run.
                            properties = self.properties(
                                self.subject_transformer.properties_of,
                                row,
                                i,
                                subject_edge_type,
                                subject_node_type,
                                node = True
                            )
                        )
                    )
                else:
                    local_errors.append(self.error(
                        f"Failed to declare subject ID for row #{i}: `{row}`.",
                        indent = 2,
                        exception = exceptions.DeclarationError ))

                # return source_node_id
                yield source_node_id

            else:
                local_errors.append(self.error(
                    f"No valid source node identifier from {self.subject_transformer} for {i}th row."
                    f" This row will be skipped.",
                    indent = 2,
                    exception = exceptions.TransformerDataError ))


    def _make_target_node_id(self, row, i, transformer, j, target_id, target_edge, target_node, local_nodes, local_errors):
        """
        Helper function to create the target node id for each target transformer.
        """

        if target_id and target_edge and target_node:

            if transformer.final_type:
                # If a final_type attribute is present in the transformer, use it as the target node type, instead
                # of the default type.
                target_node = transformer.final_type

            target_node_id = self.make_id(target_node.__name__, target_id)
            logger.debug(f"\t\tMake node {target_node_id}")
            local_nodes.append(self.make_node(node_t=target_node, id=target_node_id,
                                              # FIXME: Should we use the meta-way of accessing node properties as well?
                                              # FIXME: This would require a refactoring of the transformer interfaces and tabular.run.
                                              properties=self.properties(transformer.properties_of, row,
                                                                         i, target_edge, target_node, node=True)))

        else:
            local_errors.append(self.error(f"No valid target node identifier from {transformer}"
                                           f" for {i}th row.", indent=2, section="transformers",
                                           index=j, exception=exceptions.TransformerDataError))

            target_node_id = None

        return target_node_id

    def _make_alternative_source_node_id(self, row, i, transformer, j, target_node_id, target_edge, local_edges, local_errors):
        """
        Helper function to create an alternative source node id, in case the target transformer has a `from_subject` attribute.
        """

        found_valid_subject = False

        for t in self.transformers:
            if transformer.from_subject == t.target_type:
                found_valid_subject = True
                for s_id, s_edge, s_node, s_reverse_edge in t(row, i):
                    if s_id and s_edge and s_node:
                        if t.final_type:
                            s_node = t.final_type
                        subject_id = s_id
                        subject_node_id = self.make_id(t.target_type, subject_id)
                        logger.debug(
                            f"\t\tMake edge from {subject_node_id} toward {target_node_id}")
                        local_edges.append(
                            self.make_edge(edge_t=target_edge, id_source=subject_node_id,
                                           id_target=target_node_id,
                                           properties=self.properties(target_edge.fields(),
                                                                      row, i, s_edge, s_node)))

                    else:
                        local_errors.append(self.error(
                            f"No valid identifiers from {t} for {i}th row, when trying to change default subject type",
                            f"by {transformer} with `from_subject` attribute.",
                            indent=7, section="transformers", index=j,
                            exception=exceptions.TransformerDataError))

            else:
                # The transformer instance type does not match the type in the `from_subject` attribute.
                continue

        if not found_valid_subject:
            local_errors.append(self.error(f"\t\t\tInvalid subject declared from {transformer}."
                                           f" The subject you declared in the `from_subject` directive: `"
                                           f"{transformer.from_subject}` must not be the same as the default subject type.",
                                           exception=exceptions.ConfigError))


    def _no_element(self, local_nodes, local_edges, local_errors, local_rows, local_transformations, local_nb_nodes):
        if local_nodes == None or local_edges == None or local_errors == None \
            or local_rows == None or local_transformations == None or local_nb_nodes == None:
            logger.debug(f"There's a None in: {local_nodes}, {local_edges}, {local_errors}, {local_rows}, {local_transformations}, {local_nb_nodes}")
            return True
        return False


    def _run_all(self, process_row, nb_rows, nb_transformations, nb_nodes):
        """
        Perform the final logging after processing all rows.
        """

        if self.parallel_mapping > 0:
            logger.info(f"Processing dataframe in parallel. Number of workers set to: {self.parallel_mapping} ...")
            # Process the dataset in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                # Map the process_row function across the dataframe
                results = list(executor.map(process_row, self.iterate() ))

            # Append the results in a thread-safe manner after all rows have been processed
            i = 0
            for local_nodes, local_edges, local_errors, local_rows, local_transformations, local_nb_nodes in results:
                i += 1
                if self._no_element(local_nodes, local_edges, local_errors, local_rows, local_transformations, local_nb_nodes):
                    # logger.warning(f"Processing row {i} led to no viable element, I'll just skip it.")
                    self.non_viable_rows.add(i)
                    continue
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

            assert len(self.nodes) > 0

        elif self.parallel_mapping == 0:
            logger.debug("Processing data sequentially...")
            if self.progress_bar:
                with alive_bar(len(self), file=sys.stderr) as progress:
                    for i, row in self.iterate():
                        local_nodes, local_edges, local_errors, local_rows, local_transformations, local_nb_nodes = process_row((i, row))
                        if self._no_element(local_nodes, local_edges, local_errors, local_rows, local_transformations, local_nb_nodes):
                            # logger.warning(f"Processing row {i} led to no viable element, I'll just skip it.")
                            self.non_viable_rows.add(i)
                            continue
                        self.nodes_append(local_nodes)
                        self.edges_append(local_edges)
                        self.errors += local_errors
                        nb_rows += local_rows
                        nb_transformations += local_transformations
                        nb_nodes += local_nb_nodes
                        yield local_nodes, local_edges
                        progress()

            else: # FIXME refactor to avoid duplicated code?
                for i, row in self.iterate():
                    local_nodes, local_edges, local_errors, local_rows, local_transformations, local_nb_nodes = process_row((i, row))
                    if self._no_element(local_nodes, local_edges, local_errors, local_rows, local_transformations, local_nb_nodes):
                        # logger.warning(f"Processing row {i} led to no viable element, I'll just skip it.")
                        self.non_viable_rows.add(i)
                        continue
                    self.nodes_append(local_nodes)
                    self.edges_append(local_edges)
                    self.errors += local_errors
                    nb_rows += local_rows
                    nb_transformations += local_transformations
                    nb_nodes += local_nb_nodes
                    yield local_nodes, local_edges

        else:
            self.error(f"Invalid value for `parallel_mapping` ({self.parallel_mapping})."
                       f" Pass 0 for sequential processing, or the number of workers for parallel processing.", exception = exceptions.ConfigError)

        # Final logger
        error_count = {}
        for transformer in chain([self.subject_transformer], self.transformers, self.property_transformers):
            if transformer.output_validator:
                for msg,err in transformer.output_validator.messages.items():
                    desc = f"in {err['section']} with {transformer}: {msg}"
                    # FIXME do we want the number of validation errors or the number of bad cell values?
                    # error_count[desc] = error_count.get(desc, 0) + err['count']
                    error_count[desc] = error_count.get(desc, err['count'])
        for desc,count in error_count.items():
            logger.error(f"Recorded {count} times a validation error {desc}")

        if self.errors:
            logger.error(
                f"Recorded {len(self.errors)} errors while processing {nb_transformations} transformations with {1+len(self.transformers)} node transformers, producing {nb_nodes} nodes for {nb_rows} rows.")
            # logger.debug("\n".join(self.errors))
        else:
            logger.info(
                f"Performed {nb_transformations} transformations with {1+len(self.transformers)} node transformers, producing {nb_nodes} nodes for {nb_rows} rows.")

    # =============
    # Run function
    # =============

    def run(self):
        """Iterate through dataframe in parallel and map cell values according to YAML file, using a list of transformers."""

        logger.debug("Run...")
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
            local_transformations = 0 # Count of transformations for this row. Does not include the property transformers.
            local_nb_nodes = 0

            logger.debug(f"Process row {i}...")
            local_rows += 1

            # source_node_id = self._make_default_source_node_id(row, i, local_nodes, local_errors)
            # if not source_node_id:
            #     return None, None, None, None, None, None
            for source_node_id in self._make_default_source_node_id(row, i, local_nodes, local_errors):
                logger.debug(f"Got subject node id: `{source_node_id}`")
                if source_node_id:
                    local_nb_nodes += 1
                else:
                    return None, None, None, None, None, None

                # Loop over list of transformer instances and label_maker corresponding nodes and edges.
                # FIXME the transformer variable here shadows the transformer module.
                for j,transformer in enumerate(self.transformers):
                    local_transformations += 1
                    logger.debug(f"\tCalling the {j}th transformer: {transformer}...")
                    k = 0
                    try:
                        for target_id, target_edge, target_node, reverse_relation in transformer(row, i):
                            logger.debug(f"\t\t{k}th element yielded by transformer")
                            k += 1
                            target_node_id = self._make_target_node_id(
                                row,
                                i,
                                transformer,
                                j,
                                target_id,
                                target_edge,
                                target_node,
                                local_nodes,
                                local_errors
                            )

                            #If no valid target node id was created, an error is logged in the `_make_target_node_id` function,
                            #and we move to the next iteration of the loop.
                            if target_node_id is None:
                                continue
                            else:
                                local_nb_nodes += 1

                                # If a `from_subject` attribute is present in the transformer, loop over the transformer
                                # list to find the transformer instance mapping to the correct type, and then label_maker new
                                # subject id.

                                # FIXME add hook functions to be overloaded.

                                # FIXME: Make from_subject reference a list of subjects instead of using the add_edge function.

                                if hasattr(transformer, "from_subject"):

                                    self._make_alternative_source_node_id(
                                        row,
                                        i,
                                        transformer,
                                        j,
                                        target_node_id,
                                        target_edge,
                                        local_edges,
                                        local_errors
                                    )

                                else: # no attribute `from_subject` in `transformer`
                                    logger.debug(f"\t\tMake edge {target_edge.__name__} from {source_node_id} toward {target_node_id}")
                                    local_edges.append(
                                        self.make_edge(
                                            edge_t=target_edge,
                                            id_target=target_node_id,
                                            id_source=source_node_id,
                                            properties=self.properties(
                                                target_edge.fields(),
                                                row,
                                                i,
                                                target_edge,
                                                target_node
                                            )
                                        )
                                    )

                                    if reverse_relation:
                                        logger.info(f"\t\t\tMake reverse edge {reverse_relation.__name__} from {target_node_id} to {source_node_id}")
                                        local_edges.append(
                                            self.make_edge(
                                                edge_t=reverse_relation,
                                                id_target=source_node_id,
                                                id_source=target_node_id,
                                                properties=self.properties(
                                                    reverse_relation.fields(),
                                                    row,
                                                    i,
                                                    reverse_relation,
                                                    source_node_id.__class__
                                                )
                                            )
                                        )
                    except Exception as err:
                        logger.error(f"Error while calling the {j}th transformer on the {i}th row, after having yielded {k} items.")
                        raise err
            # assert hasattr(local_nodes, "__iter__")
            # assert hasattr(local_edges, "__iter__")
            return local_nodes, local_edges, local_errors, local_rows, local_transformations, local_nb_nodes
            # End of process_row local function

        nb_rows = 0
        nb_transformations = 0
        nb_nodes = 0

        if self.parallel_mapping > 0:
            logger.debug("\tParallel mapping...")
            self._run_all(process_row, nb_rows, nb_transformations, nb_nodes)
        else:
            logger.debug("\tSequential mapping...")
            for local_nodes,local_edges in self._run_all(process_row, nb_rows, nb_transformations, nb_nodes):
                yield local_nodes, local_edges

    def __call__(self):
        # FIXME If the run functions contains a generator nested under an `if`, the call function is note even called by the derivated class instance..."
        # if self.parallel_mapping > 0:
        #     logger.debug("Parallel mapping")
        #     self.run()
        #     return
        # else:
        #     logger.debug("Sequential mapping")
        for local_nodes, local_edges in self.run():
            yield local_nodes, local_edges

    @abstractmethod
    def iterate(self):
        """ The function that access the atomic item of its targeted iterable data structure.

            It is called by run() in a way that is equivalent to:

            .. code-block:: python

                for i,item in self.iterate():

            For example, for a table, it yield the row number and the row itself.

            Returns:
                The index of the item and the item.
        """
        raise NotImplementedError


    def __len__(self):
        """ Return the (estimated) len of the data that the Adapter manages.

            Returns:
                An integer, or None if length is undefined.
        """
        return None


    def __del__(self):
        if self.non_viable_rows:
            logger.warning( \
                f"Got {len(self.non_viable_rows)} rows that produced no valid element." \
                " I skipped them silently, but you may want to double check your mapping." \
                " This is normal if you used a match section that does not consider" \
                " all possible values. Run with the DEBUG log level to see the row numbers.")
            logger.debug(", ".join(str(i) for i in self.non_viable_rows))

