""" Classes related to loading input data.
"""

import sys
import glob
import rdflib
import pathlib
import logging

import json as pyjson
import pandas as pd

from alive_progress import alive_bar
from abc import ABCMeta as ABSTRACT, abstractmethod

from . import tabular
from . import owl
from . import xml
from . import json
from . import exceptions

logger = logging.getLogger("ontoweaver")


class Loader(metaclass = ABSTRACT):
    """ The interface for classes that load data.

        Classes implementing this interface are in charge of checking if they can
        load the given data, and indicate which base.Adapter can manage what they load.
    """

    def __init__(self, progress_bar = False):
        self.progress_bar = progress_bar

    def __call__(self, data, **kwargs):
        """If the loader allows the passed data, then load and return them.

           This will pass any named argument to the load() function.

           Args:
               data: the data to be loaded.
               kwargs: any named argument dictionary

           Returns:
               The loaded data structure.
        """
        if self.allows(data):
            return self.load(data, **kwargs)

    @abstractmethod
    def allows(self, data):
        """ Returns True if the given data can be loaded by this Loader.

            The given data can be string refereing to a filename,
            or any typed data structure.

            Args:
                data: the data to be checked

            Returns:
                True, if the given data is allowed, else False.
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, data : list, **kwargs):
        """ Load the data structure from the given item.

            The given item can be a string representing a filename,
            or a typed data structure. In the later case, this method
            will probably just return it as-is.

            Args:
                data: the list of data items to load
                kwargs: any named argument that this method may use or pass to subfunctions
        """
        raise NotImplementedError()

    @abstractmethod
    def adapter(self, **kwargs):
        """ Return the type of the adapter that can process the loaded data.

            Args:
                kwargs: any named argument that this method may use or pass
        """
        return NotImplementedError()

    def extensions(self, filenames):
        """ Return the extensions of the given filenames

            Utility function for loaders interested in filenames.

            Args:
                filenames: a list of filenames (strings or pathlid.Path objects)

            Returns:
                A list of extensions, with their dot.
        """
        assert type(filenames) == list, "You must pass a list to this function"

        if not all(type(f) == str for f in filenames):
            # Those cannot be extensions
            return None

        extensions = []
        for filename in filenames:
            if type(filename) == str or type(filename) == pathlib.Path:
                ext = pathlib.Path(filename).suffix
                if not ext:
                    # If the passed filename is itself an extension,
                    # just use it as-is.
                    ext = filename
                assert ext != '', f"I can't parse the extension of file {filename}"
                extensions.append(ext)
            else:
                logger.warning(f"I don't know how to handle the filename `{filename}` of type `{type(filename)}`. I'll pretend I saw nothing, but this may generate errors later on.")
        return extensions


class LoadPandasDataframe(Loader):
    """ Load Pandas DataFrame

        This mainly concatenates the given DataFrames into a single one.
        Pandas will raises an error if that's not possible.
    """
    def allows(self, data):
        return all(type(d) == pd.DataFrame for d in data)

    def load(self, dataframes, **kwargs):
        logger.debug(dataframes)
        assert type(dataframes) != str and len(dataframes) > 0 and not any(d.empty for d in dataframes), "A Loader expects a list (or an iterable) of data."
        return pd.concat(dataframes)

    def adapter(self, **kwargs):
        return tabular.PandasAdapter


class LoadPandasFile(Loader):
    """Read a file with Pandas, using its extension to guess its format.

    If no additional arguments are passed, it will call the
    Pandas `read_*` function with `filter_na = False`, which makes empty cell
    values to be loaded as empty strings instead of NaN values.

    Args:
        filename: The name of the data file the user wants to map.
        separator (str): The separator used in the data file. Defaults to None.
        kwargs: A dictionary of arguments to pass to pandas.read_* functions.

    Raises:
        exceptions.FeatureError: if the extension is unknown.

    Returns:
        A Pandas DataFrame.
    """

    def __init__(self, progress_bar = False):
            # We probably don't want NaN as a default,
            # since they tend to end up in a label.
            #'c' engine does not support regex separators (separators > 1 char and different
            # from '\s+' are interpreted as regex) which results in an error.
        self.read_funcs = {
            '.csv'    : (self.read_csv_progress, {
                'sep': ',',
                'na_filter': True, # Do not load empty cells, if possible.
                'dtype': str, # Always load data as string, to avoid conversion of numbers to floating-point values with decimal separators.
                'engine': 'python'
            }),
            '.tsv'    : (self.read_csv_progress, {
                'sep': '\t',
                'na_filter': True,
                'dtype': str,
                'engine': 'python'
            }),
            '.txt'    : (self.read_csv_progress, {
                'na_filter': True,
                'dtype': str,
                'engine': 'python'
            }),
            '.dat'    : (self.read_csv_progress, {
                'na_filter': True,
                'dtype': str,
                'engine': 'python'
            }),

            # Use dtype: str for formats not supposed to host complex objects in their cells.
            '.xls'    : (pd.read_excel,   {'na_filter': True, 'dtype': str}),
            '.xlsx'   : (pd.read_excel,   {'na_filter': True, 'dtype': str}),
            '.xlsm'   : (pd.read_excel,   {'na_filter': True, 'dtype': str}),
            '.xlsb'   : (pd.read_excel,   {'na_filter': True, 'dtype': str}),
            '.odf'    : (pd.read_excel,   {'na_filter': True, 'dtype': str}),
            '.ods'    : (pd.read_excel,   {'na_filter': True, 'dtype': str}),
            '.odt'    : (pd.read_excel,   {'na_filter': True, 'dtype': str}),
            '.orc'    : (pd.read_orc,     {'na_filter': True, 'dtype': str}),
            '.sas'    : (pd.read_sas,     {'na_filter': True, 'dtype': str}),
            '.spss'   : (pd.read_spss,    {'na_filter': True, 'dtype': str}),
            '.stata'  : (pd.read_stata,   {'na_filter': True, 'dtype': str}),

            # Allow multiple dtypes for the others.
            '.json'   : (pd.read_json,    {'na_filter': True}),
            '.feather': (pd.read_feather, {'na_filter': True}),
            '.pickle' : (pd.read_pickle,  {'na_filter': True}),
            '.hdf'    : (pd.read_hdf,     {'na_filter': True}),
            '.parquet': (pd.read_parquet, {}), # There's no na_filter in read_parquet.
        }

        super().__init__(progress_bar)

    def read_csv_progress(self, filename, hint=None, steps=100, estimate_lines=10, **kwargs):
        if self.progress_bar:
            del kwargs["progress_bar"]
            chunks = []
            if hint:
                nb_lines = hint

                # How many lines to read at each iteration.
                if "chunksize" not in kwargs:
                    chunksize = int(math.ceil(nb_lines / steps))

                with alive_bar(steps, file=sys.stderr) as progress:
                    for chunk in pd.read_csv(filename, chunksize=chunksize, **kwargs):
                        chunks.append(chunk)
                        progress()
            else:
                if "chunksize" not in kwargs:
                    chunksize = 100

                with alive_bar(file=sys.stderr) as progress:
                    for chunk in pd.read_csv(filename, chunksize=chunksize, **kwargs):
                        chunks.append(chunk)
                        progress()

            df = pd.concat(chunks, axis=0)

            return df
        else:
            if "progress_bar" in kwargs:
                del kwargs["progress_bar"]
            return pd.read_csv(filename, **kwargs)


    def allows(self, filenames):
        # assert type(filenames) != str and len(filenames) > 0 and all(i != '' for i in filenames), "A Loader expects a list (or an iterable) of data."

        exts = self.extensions(filenames)
        if not exts:
            return False

        if all(e in self.read_funcs.keys() for e in exts):
            return True
        else:
            return False


    def load(self, filenames, **kwargs):
        assert type(filenames) != str and len(filenames) > 0 and all(i != '' for i in filenames), "A Loader expects a list (or an iterable) of data."

        exts = list(set(self.extensions(filenames)))
        assert exts
        if not self.allows(exts):
            msg = f"One of those file formats: `{', '.join(exts)}` is not supported (I can only read one of: {', '.join(self.read_funcs.keys())})"
            logger.error(msg)
            raise exceptions.FeatureError(msg)

        expanded = []
        for filename in filenames:
            globbed = glob.glob(filename)
            expanded += globbed
        filenames = expanded

        data = []
        for filename in filenames:
            loadfunc  = self.read_funcs[pathlib.Path(filename).suffix][0]
            kw = self.read_funcs[pathlib.Path(filename).suffix][1]

            # Overwrite default named arguments with the passed ones.
            kw.update(kwargs)

            if not kw:
                # pd.read_parquet does not allow even an empty kwargs.
                logger.debug(f"No argument passed to `{pathlib.Path(filename).suffix}` load function.")
                try:
                    data.append( loadfunc(filename) )
                except AttributeError as e:
                    logger.error(f"I cannot load those parquet files. Maybe they don't have the 'name' property? That may explain the following error in Pandas.")
                    raise e
            else:
                logger.debug(f"Additional arguments passed to the `{pathlib.Path(filename).suffix}` load function: {kw}")
                data.append( loadfunc(filename, **kw) )

        if not data:
            # self.error(f"I failed to load data from files: {', '.join()}", section = "data loading", exception = exceptions.InputDataError)
            msg = f"I failed to load data from files: {', '.join(filenames)}"
            raise exceptions.InputDataError(msg)

        out = pd.concat(data)
        return out

    def adapter(self, **kwargs):
        return tabular.PandasAdapter


class LoadOWLGraph(Loader):
    """ Load the given rdflib.Graph(s)

        If adapter() function is given the string "automap", this will select the
        owl.OWLAutoAdapter, else the owl.OWLAdapter.

        If you try to load several graphs, this will show a warning stating
        that it is quite improbable that this would work.
    """
    def allows(self, data):
        return all(type(d) == rdflib.Graph for d in data)

    def load(self, graphs, **kwargs):
        assert type(graphs) != str and len(graphs) > 0 and all(i != '' for i in graphs), "A Loader expects a list (or an iterable) of data."
        if len(graphs) > 1:
            logger.warning(
                "Loading multiple OWL graphs at once does not guarantee a correct fusion." \
                " That is, graph operations in RDFLib are assumed to be performed in subgraphs" \
                " of some larger database and assume shared blank node IDs, and therefore may" \
                " cause unwanted collisions of blank-nodes in graph (cf." \
                " https://rdflib.readthedocs.io/en/7.1.1/merging.html)." \
                " To avoid this warning, either load only compatible OWL graphs at once" \
                " either, load OWL graphs separately and" \
                " use OntoWeaver's fusion features afterward.")

        graph = rdflib.Graph()
        for g in graphs:
            graph += g

        return graph

    def adapter(self, **kwargs):
        if "automap" in kwargs:
            logger.debug("Asked for `automap`, I'll use owl.OWLAutoAdapter")
            return owl.OWLAutoAdapter
        else:
            logger.debug("Asked for regular mapping, I'll use owl.OWLAdapter")
            return owl.OWLAdapter


class LoadOWLFile(Loader):
    """ Load the given OWL file.

        If adapter() function is given the string "automap", this will select the
        owl.OWLAutoAdapter, else the owl.OWLAdapter.

        If you try to load several graphs, this will show a warning stating
        that it is quite improbable that this would work.
    """
    def __init__(self, progress_bar = False):
        self.allowed = [".owl", ".n3", ".turtle", ".ttl", ".nt", ".trig", ".trix", ".json-ld"]
        super().__init__(progress_bar)

    def allows(self, filenames):
        # assert type(filenames) != str and len(filenames) > 0 and all(i != '' for i in filenames), "A Loader expects a list (or an iterable) of data."

        exts = self.extensions(filenames)
        if not exts:
            return False

        if all(e in self.allowed for e in exts):
            return True
        else:
            return False

    def load(self, filenames, **kwargs):
        assert type(filenames) != str and len(filenames) > 0 and all(i != '' for i in filenames), "A Loader expects a list (or an iterable) of data."
        if len(filenames) > 1:
            logger.warning(
                "Loading multiple OWL files at once does not guarantee a correct fusion." \
                " That is, graph operations in RDFLib are assumed to be performed in subgraphs" \
                " of some larger database and assume shared blank node IDs, and therefore may" \
                " cause unwanted collisions of blank-nodes in graph (cf." \
                " https://rdflib.readthedocs.io/en/7.1.1/merging.html)." \
                " To avoid this warning, either load only compatible OWL files at once" \
                " either, load OWL files separately and" \
                " use OntoWeaver's fusion features afterward.")

        exts = set(self.extensions(filenames))
        assert exts
        for ext in exts:
            if not self.allows([ext]):
                msg = f"File format '{ext}' is not supported (I can only read one of: {' ,'.join(self.read_funcs.keys())})"
                logger.error(msg)
                raise exceptions.FeatureError(msg)

        g = rdflib.Graph()
        for filename in filenames:
            ext = pathlib.Path(filename).suffix
            if ext == ".owl":
                g.parse(filename, format = "xml")
            else:
                g.parse(filename) # Guess the format based on extension.

        return g


    def adapter(self, **kwargs):
        if "automap" in kwargs:
            logger.debug("Asked for `automap`, I'll use owl.OWLAutoAdapter")
            return owl.OWLAutoAdapter
        else:
            logger.debug("Asked for regular mapping, I'll use owl.OWLAdapter")
            return owl.OWLAdapter


class LoadXMLString(Loader):
    """ Load (here, that is: parse) the given XML string.

        If you try to load several graphs, this will show a warning stating
        that it is quite improbable that this would work.
    """
    def allows(self, data):
        # assert type(data) != str and len(data) > 0 and all(i != '' for i in data), "A Loader expects a list (or an iterable) of data."
        return all(type(d) == str for d in data)

    def load(self, xmls, **kwargs):
        assert type(xmls) != str and len(xmls) > 0 and all(i != '' for i in xmls), "A Loader expects a list (or an iterable) of xmls."
        if len(xmls) > 1:
            logger.warning(
                "Loading multiple XML files at once is not really supported," \
                " since I'm only going to concatenate the documents." \
                " I thus cannot guarantee a well-formed XML document at the end." \
                " Ignore this warning only if you know how to sequentially assemble a valid XML document.")

        return "\n".join(xmls)

    def adapter(self, **kwargs):
        return xml.XMLAdapter


class LoadXMLFile(Loader):
    """ Load the given XML file.

        If you try to load several graphs, this will show a warning stating
        that it is quite improbable that this would work.
    """
    def __init__(self, progress_bar = False):
        self.xl = LoadXMLString()
        self.allowed = [".xml"]
        super().__init__(progress_bar)

    def allows(self, filenames):
        # assert type(data) != str and len(data) > 0 and all(i != '' for i in data), "A Loader expects a list (or an iterable) of data."

        exts = self.extensions(filenames)
        if not exts:
            return False

        if all(e in self.allowed for e in exts):
            return True
        else:
            return False

    def load(self, filenames, **kwargs):
        assert type(filenames) != str and len(filenames) > 0 and all(i != '' for i in filenames), "A Loader expects a list (or an iterable) of filenames."
        if len(filenames) > 1:
            logger.warning(
                "Loading multiple XML files at once is not really supported," \
                " since I'm only going to concatenate the documents." \
                " I thus cannot guarantee a well-formed XML document at the end." \
                " Ignore this warning only if you know how to sequentially assemble a valid XML document.")

        data = []
        for filename in filenames:
            ext = pathlib.Path(filename).suffix
            if not self.allows([filename]):
                msg = f"File format '{ext}' of file '{filename}' is not supported (I can only read one of: {' ,'.join(self.allowed)})"
                logger.error(msg)
                raise exceptions.FeatureError(msg)

            with open(filename, 'r') as fd:
                data.append( fd.read() )

        return self.xl(data)

    def adapter(self, **kwargs):
        return xml.XMLAdapter


class LoadJSONString(Loader):
    """ Load (here, that is: parse) the given JSON string.
    """
    def allows(self, jsons):
        # assert type(jsons) != str and len(jsons) > 0 and all(i != '' for i in jsons), "A Loader expects a list (or an iterable) of data."
        if not all(type(j) == str for j in jsons):
            return False

        for j in jsons:
            try:
                pyjson.loads(j)
            except ValueError as e:
                return False

        return True

    def load(self, jsons, **kwargs):
        dic = {}
        logger.debug("Assembling JSON strings into a single data structure.")
        for j in jsons:
            d = pyjson.loads(j)
            for k in d:
                if k in dic:
                    logger.warning(
                        f"Overwriting `{k}:{dic[k]}` with `{d[k]}` in a previously loaded JSON." \
                        " This happens if you ask to load multiple non-independant JSONs." \
                        " You may also want to double-check that the JSON list has the order that you want." \
                    )
            dic.update(d)
        return pyjson.dumps(dic)

    def adapter(self, **kwargs):
        return json.JSONAdapter


class LoadJSONFile(Loader):
    """ Load the given JSON file.
    """
    def __init__(self, progress_bar = False):
        self.allowed = [".json"]
        self.jl = LoadJSONString()
        super().__init__(progress_bar)

    def allows(self, filenames):
        # assert type(filenames) != str and len(filenames) > 0 and all(i != '' for i in filenames), "A Loader expects a list (or an iterable) of data."
        return all(pathlib.Path(f).suffix in self.allowed for f in filenames)

    def load(self, filenames, **kwargs):
        assert type(filenames) != str and len(filenames) > 0 and all(i != '' for i in filenames), "A Loader expects a list (or an iterable) of data."

        jsons = []
        for filename in filenames:
            ext = pathlib.Path(filename).suffix
            if not self.allows([filename]):
                msg = f"File format '{ext}' of file '{filename}' is not supported (I can only read one of: {' ,'.join(self.allowed)})"
                logger.error(msg)
                raise exceptions.FeatureError(msg)

            with open(filename, 'r') as fd:
                jsons.append( fd.read() )

        return self.jl.load(jsons)

    def adapter(self, **kwargs):
        return json.JSONAdapter

