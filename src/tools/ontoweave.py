#!/usr/bin/env python3

if __name__ == "__main__":
    import os
    import sys
    import logging
    import argparse
    import subprocess

    name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    do = argparse.ArgumentParser(
        description = "A command line tool to run OntoWeaver mapping adapters on a set of tabular data, and call the created BioCypher export scripts.",
        epilog = f"Example usage:\n  {name} table.csv:mapping.yaml\n  {name} --config biocypher_config.yaml --schema schema.yaml table1.csv:mapping1.yaml table2.tsv:mapping1.yaml table3.parquet:mapping2.yaml --import-script-run",
        formatter_class = argparse.RawTextHelpFormatter)

    # {name} --map data_stuff.csv:stuff_mapping.yaml --map data_stiff.csv:stiff_mapping.yaml --config biocypher_config.yaml --schema schema.yaml --call-import-script

    do.add_argument("mapping", metavar="FILE:MAPPING", nargs="+",
        help=f"Run the given YAML MAPPING to extract data from the tabular FILE (usually a CSV). Several mappings can be passed to {name}. You may also use the same mapping on different data files.")

    do.add_argument("-c", "--config", metavar="FILE", default="biocypher_config.yaml",
        help="The BioCypher config file (the one managing ontologies and the output backend). [default: %(default)s]")

    do.add_argument("-s", "--schema", metavar="FILE", default="schema.yaml",
        help="The BioCypher schema file (the one managing node and edge types). [default: %(default)s]")

    do.add_argument("-i", "--import-script-run", action="store_true",
        help=f"If passed {name} will call the import scripts created by Biocypher for you.")

    do.add_argument("-S", "--prop-sep", metavar="CHARACTER", default = ";",
        help="The character used to separate property values fusionned in the same property at the reconciliation step. [default: %(default)s]")

    do.add_argument("-a", "--type-affix", choices=["suffix","prefix","none"], default="none",
        help="Where to add the type string to the ID label. [default: %(default)s]")

    do.add_argument("-A", "--type-affix-sep", metavar="CHARACTER", default=":",
        help="Character used to separate the label from the type affix. [default: %(default)s]")

    do.add_argument("-l", "--log-level", default="WARNING",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Configure the log level. [default: %(default)s]")

    asked = do.parse_args()

    logging.basicConfig(level = asked.log_level)

    logging.info("OntoWeave parameters:")
    logging.info("\tmappings:")
    mappings = {}
    for data_map in asked.mapping:
        data,map = data_map.split(":")
        mappings[data] = map
        logging.info(f"\t\t`{data}` => `{map}`")

    logging.info(f"\tconfig: `{asked.config}`")
    logging.info(f"\tschema: `{asked.schema}`")
    logging.info(f"\tprop-sep: `{asked.prop_sep}`")
    logging.info(f"\ttype-affix: `{asked.type_affix}`")
    logging.info(f"\ttype-affix-sep: `{asked.type_affix_sep}`")
    logging.info(f"\timport-script-run: `{asked.import_script_run}`")
    logging.info(f"\tlog-level: `{asked.log_level}`")

    # Late import to avoid useless Biocypher's logs when asking for --help.
    import ontoweaver

    logging.info(f"Running OntoWeaver...")
    import_file = ontoweaver.extract_reconciliate_write(asked.config, asked.schema, mappings, separator=asked.prop_sep, affix=asked.type_affix, affix_separator = asked.type_affix_sep)

    # Output import file on stdout, in case the user would want to capture it.
    print(import_file)

    if asked.import_script_run:
        logging.info(f"Run import scripts...")
        subprocess.run(import_script)

    logging.info("Done")
