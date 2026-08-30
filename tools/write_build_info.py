#!/usr/bin/env python

import argparse
import json
from pathlib import Path

from build_recipe import PACKAGE_FILES, recipe_hash


def write_build_info(build, output):
    build = {**build, 'recipe': recipe_hash(build['name'])}
    Path(output).write_text(f'{json.dumps(build, indent=2)}\n', encoding='utf-8')
    return build


def main():
    parser = argparse.ArgumentParser(description='Write wheel build metadata')
    parser.add_argument('package', choices=sorted(PACKAGE_FILES))
    parser.add_argument('--version', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--os', required=True)
    parser.add_argument('--mkl', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    write_build_info(
        {
            'name': args.package,
            'version': args.version,
            'python': args.python,
            'os': args.os,
            'mkl': args.mkl,
        },
        args.output,
    )


if __name__ == '__main__':
    main()
