#!/usr/bin/env python

import argparse
import contextlib
import json
from pathlib import Path

with contextlib.suppress(ImportError):
    from packaging.version import Version


class Build:
    def __init__(self, info):
        if isinstance(info, dict):
            self.info = info
        else:
            self.info = json.loads(info.read_text())

        self.name = self.info['name'].replace('-', '_')
        self.version = self.info['version']
        self.python = self.info['python']
        self.os = self.info['os'].lower()
        self.mkl = self.info['mkl']
        self.recipe = self.info.get('recipe')

        self.key = '-'.join([self.name, self.version, self.python, self.os])

    def exclude(self, store, check_mkl=False):
        if self.key not in store:
            return False

        stored = store[self.key]
        bump_mkl = check_mkl and Version(self.mkl) > Version(stored['mkl'])
        stale_recipe = self.recipe is not None and stored.get('recipe') != self.recipe
        return not bump_mkl and not stale_recipe

    def merge_with(self, store):
        if self.recipe is None:
            raise ValueError('Build metadata is missing its recipe digest')
        store[self.key] = {'mkl': self.mkl, 'recipe': self.recipe}


def fetch_store(path):
    if isinstance(path, str):
        path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def main(storepath, builds):
    store = fetch_store(storepath)
    for b in builds:
        Build(b).merge_with(store)

    with storepath.open('w') as f:
        json.dump(store, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Store build meta data')
    parser.add_argument('builds', nargs='+', help='input file(s)')
    parser.add_argument('-s', '--store', required=True, type=str, help='output file')

    args = parser.parse_args()
    store = Path(args.store)
    builds = [Path(f) for f in args.builds]

    main(store, builds)
