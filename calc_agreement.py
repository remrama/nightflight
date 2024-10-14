"""
Evaluate agreement between islucid coders.
"""

import pandas as pd
from sklearn import metrics

import utils


# Pick export path names
export_path = utils.derivatives_directory / "agreement.tsv"

# Load human codings
coder1 = utils.load_codings("annotator1")
coder2 = utils.load_codings("annotator2")

# Generate a dataframe of agreement metrics
accuracy = metrics.accuracy_score(coder1, coder2)
kappa = metrics.cohen_kappa_score(coder1, coder2)
agreement = pd.DataFrame(
    [{"metric": "accuracy", "value": accuracy}, {"metric": "kappa", "value": kappa}]
)

# Save agreement metrics
agreement.to_csv(export_path, index=False, sep="\t", encoding="utf-8")
