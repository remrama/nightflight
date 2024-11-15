"""
Validate the exported archive.
"""
from contextlib import chdir
from frictionless import Package

import config


archive_dir = config.directories["archive"]
datapackage_path = "datapackage.json"

# This will catch with a FrictionlessException if the datapackage.json is not valid.

with chdir(archive_dir):
    report = Package("datapackage.json").validate()

if not report.valid:
    print(report.to_summary())
else:
    for task in report.tasks:
        if task.valid:
            name = task.name
            path = task.place
            stats = task.stats
            hash_ = stats["hash"]
            bytes_ = stats["bytes"]

# version = report["version"]

# from pandera.io import from_frictionless_schema
# for resource in data["resources"]:
#     frictionless_schema = resource["schema"]
#     pandera_schema = from_frictionless_schema(frictionless_schema)


