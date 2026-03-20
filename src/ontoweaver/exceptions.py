"""
The exceptions hierarchy of what OntoWeaver may raise.

Classes in this file all have a static ``code`` variable,
which is used by the ``ontoweave`` command as an error return code.

The exceptions taxonomy is::

    OntoWeaverError
    ├─ AutoSchemaError
    ├─ ConfigError
    ├─ FeatureError
    ├─ ParsingError
    │  ├─ ParsingDeclarationsError
    │  │  ├─ CardinalityError
    │  │  └─ MissingDataError
    │  ├─ # ParsingPropertiesError
    │  │  └─ # MissingFieldError
    └─ RunError
       ├─ DeclarationError
       ├─ FileError
       │  ├─ FileAccessError
       │  └─ FileOverwriteError
       ├─ InputDataError
       ├─ SubprocessError
       └─ TransformerError
          ├─ DataValidationError
          ├─ InterfaceInheritanceError
          ├─ TransformerConfigError
          ├─ TransformerDataError
          ├─ TransformerInputError
          └─ TransformerInterfaceError
"""

class OntoWeaverError(Exception):
    """The base of all errors related to OntoWeaver."""
    code = 255

class AutoSchemaError(OntoWeaverError):
    """Your mapping is inconsistent with the ontology's taxonomy."""
    code = 79

class ConfigError(OntoWeaverError):
    """Your configuration has an error."""
    code = 78  # "bad config"

class FeatureError(OntoWeaverError):
    """This feature does not seem to exist."""
    code = 64  # alt "usage"

class ParsingError(OntoWeaverError):
    """I failed to parse a YAML file."""
    code = 65  # "data format"

class ParsingDeclarationsError(ParsingError):
    """Your mapping YAML file has an error."""
    code = 85

class CardinalityError(ParsingDeclarationsError):
    """Your mapping YAML file shows duplicated parameters."""
    code = 86

class MissingDataError(ParsingDeclarationsError):
    """Your mapping YAML file has missing parameter(s)."""
    code = 66  # "no input"

# class ParsingPropertiesError(ParsingError):

# class MissingFieldError(ParsingPropertiesError):

class RunError(OntoWeaverError):
    """Error while running OntoWeaver."""
    code = 70  # "internal"

class DeclarationError(RunError):
    """I cannot run your mapping, it has logical inconsistency."""
    code = 87

class FileError(RunError):
    """I have a problem with a file."""
    code = 72  # "OS file"

class FileAccessError(FileError):
    """I cannot read or write in a file."""
    code = 77  # "no perm"

class FileOverwriteError(FileError):
    """I cannot overwrite a file."""
    code = 126  # alt "no perm"

class InputDataError(RunError):
    """Some input data is inconsistent."""
    code = 132  # "illegal"

class TransformerError(RunError):
    """A transformer cannot run properly."""
    code = 200

class SubprocessError(RunError):
    """A subprocess ended on an error."""
    code = 128  # "bad exit"

class DataValidationError(TransformerError):
    """Some data validation do not pass."""
    code = 76  # "protocol"

class InterfaceInheritanceError(TransformerError):
    """The declaration of your user-made transformer is wrong."""
    code = 201

class TransformerConfigError(TransformerError):
    """Some parameters of your transformer are inconsistent."""
    code = 202

class TransformerDataError(TransformerError):
    """A transformer cannot process its data."""
    code = 203

class TransformerInputError(TransformerError):
    """The data a transformer transforms is inconsistent."""
    code = 204

class TransformerInterfaceError(TransformerError):
    """You used a transformer the wrong way."""
    code = 205

