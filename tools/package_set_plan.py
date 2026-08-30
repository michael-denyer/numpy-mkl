#!/usr/bin/env python

import argparse
import json
from pathlib import Path

PACKAGES = ('mkl-service', 'numpy', 'scipy')
COORDINATE_FIELDS = ('python_version', 'runner', 'container', 'os')


class PackageSetPlanError(ValueError):
    pass


def matrix_entries(package, field):
    matrix = package.get(field)
    return [] if matrix is None else matrix['include']


def coordinate(entry):
    return tuple(entry.get(field) for field in COORDINATE_FIELDS)


def build_plan(packages, force=False):
    if set(packages) != set(PACKAGES):
        raise PackageSetPlanError(f'Expected package data for {PACKAGES!r}')

    full = {
        name: {
            coordinate(entry): entry for entry in matrix_entries(data, 'full_matrix')
        }
        for name, data in packages.items()
    }
    common = set.intersection(*(set(entries) for entries in full.values()))
    if not common:
        gaps = {name: sorted(entries) for name, entries in full.items()}
        raise PackageSetPlanError(f'Packages have no common build coordinates: {gaps}')

    for key in common:
        entries = [full[name][key] for name in PACKAGES]
        if any(entry != entries[0] for entry in entries[1:]):
            raise PackageSetPlanError(
                f'Packages disagree on build policy for coordinate {key}: {entries}'
            )

    stale = {
        coordinate(entry)
        for data in packages.values()
        for entry in matrix_entries(data, 'matrix')
    }
    selected = common if force else common & stale
    reference = full['mkl-service']
    include = [reference[key] for key in sorted(selected)]

    unsupported = {
        name: [list(key) for key in sorted(set(entries) - common)]
        for name, entries in full.items()
    }
    return {
        'packages': {
            name: {
                'version': data['version'],
                'tag': data['tag'],
            }
            for name, data in packages.items()
        },
        'matrix': {'include': include} if include else None,
        'common': [list(key) for key in sorted(common)],
        'unsupported': unsupported,
    }


def main():
    parser = argparse.ArgumentParser(description='Plan one exact wheel package set')
    for package in PACKAGES:
        parser.add_argument(f'--{package}', required=True, type=Path)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    packages = {
        package: json.loads(getattr(args, package.replace('-', '_')).read_text())
        for package in PACKAGES
    }
    print(json.dumps(build_plan(packages, force=args.force)))


if __name__ == '__main__':
    main()
