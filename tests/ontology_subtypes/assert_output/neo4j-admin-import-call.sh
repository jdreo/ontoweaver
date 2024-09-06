#!/bin/bash
version=$(bin/neo4j-admin --version | cut -d '.' -f 1)
if [[ $version -ge 5 ]]; then
	neo4j-admin database import full --delimiter=";" --array-delimiter="|" --quote="'" --overwrite-destination=true --skip-bad-relationships=true --skip-duplicate-nodes=true --nodes="/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/SequenceVariant-header.csv,/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/SequenceVariant-part.*" --nodes="/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/FdaEvidenceLevel-header.csv,/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/FdaEvidenceLevel-part.*" --nodes="/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/Patient-header.csv,/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/Patient-part.*" --nodes="/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/OncokbEvidenceLevel-header.csv,/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/OncokbEvidenceLevel-part.*" --relationships="/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/VariantToEvidence-header.csv,/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/VariantToEvidence-part.*" --relationships="/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/PatientHasVariant-header.csv,/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/PatientHasVariant-part.*" test
else
	neo4j-admin import --delimiter=";" --array-delimiter="|" --quote="'" --force=true --skip-bad-relationships=true --skip-duplicate-nodes=true --nodes="/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/SequenceVariant-header.csv,/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/SequenceVariant-part.*" --nodes="/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/FdaEvidenceLevel-header.csv,/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/FdaEvidenceLevel-part.*" --nodes="/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/Patient-header.csv,/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/Patient-part.*" --nodes="/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/OncokbEvidenceLevel-header.csv,/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/OncokbEvidenceLevel-part.*" --relationships="/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/VariantToEvidence-header.csv,/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/VariantToEvidence-part.*" --relationships="/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/PatientHasVariant-header.csv,/Users/mbaric/ontoweaver1/biocypher-out/20240711135425/PatientHasVariant-part.*" --database=test 
fi