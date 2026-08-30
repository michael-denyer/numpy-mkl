#!/usr/bin/env python

import runpy
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

ROOT = Path(__file__).parents[1]
TOOLS = ROOT / 'tools'
sys.path.insert(0, str(TOOLS))

import build_recipe  # noqa: E402
import verify_package_set  # noqa: E402
from store_info import Build  # noqa: E402


class TestBuildIdentity(unittest.TestCase):
    def build(self, recipe='sha256:current'):
        return Build(
            {
                'name': 'numpy',
                'version': '2.5.2',
                'python': 'cp312',
                'os': 'Linux',
                'mkl': '2026.1.0',
                'recipe': recipe,
            }
        )

    def test_matching_recipe_skips_build(self):
        store = {
            'numpy-2.5.2-cp312-linux': {
                'mkl': '2026.1.0',
                'recipe': 'sha256:current',
            }
        }

        self.assertTrue(self.build().exclude(store))

    def test_changed_recipe_rebuilds(self):
        store = {
            'numpy-2.5.2-cp312-linux': {
                'mkl': '2026.1.0',
                'recipe': 'sha256:old',
            }
        }

        self.assertFalse(self.build().exclude(store))

    def test_legacy_store_entry_rebuilds(self):
        store = {'numpy-2.5.2-cp312-linux': {'mkl': '2026.1.0'}}

        self.assertFalse(self.build().exclude(store))

    def test_legacy_caller_keeps_coordinate_only_behavior(self):
        store = {'numpy-2.5.2-cp312-linux': {'mkl': '2026.1.0'}}

        self.assertTrue(self.build(recipe=None).exclude(store))

    def test_force_bypasses_matching_store(self):
        fetch_matrix = runpy.run_path(str(TOOLS / 'fetch_matrix2'))
        matrix = fetch_matrix['FetchBuildMatrix'].__new__(
            fetch_matrix['FetchBuildMatrix']
        )
        matrix.config = SimpleNamespace(force_build=True)

        self.assertFalse(matrix.exclude('cp312', ('ubuntu-latest', None)))

    def test_merge_requires_and_persists_recipe(self):
        store = {}
        self.build().merge_with(store)
        self.assertEqual(
            store['numpy-2.5.2-cp312-linux'],
            {'mkl': '2026.1.0', 'recipe': 'sha256:current'},
        )

        with self.assertRaisesRegex(ValueError, 'missing its recipe digest'):
            self.build(recipe=None).merge_with(store)

    def test_digest_is_deterministic_and_package_specific(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recipe_files = {
                *build_recipe.COMMON_FILES,
                *(path for files in build_recipe.PACKAGE_FILES.values() for path in files),
            }
            for relative in recipe_files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative)

            before = {
                package: build_recipe.recipe_hash(package, root)
                for package in build_recipe.PACKAGE_FILES
            }
            (root / 'patches/numpy/init_mkl.patch').write_text('changed')
            after = {
                package: build_recipe.recipe_hash(package, root)
                for package in build_recipe.PACKAGE_FILES
            }

        self.assertRegex(before['numpy'], r'^sha256:[0-9a-f]{64}$')
        self.assertNotEqual(before['numpy'], after['numpy'])
        self.assertEqual(before['mkl-service'], after['mkl-service'])
        self.assertEqual(before['scipy'], after['scipy'])


class TestWorkflowContracts(unittest.TestCase):
    def workflow(self, name):
        return (ROOT / '.github/workflows' / name).read_text()

    def test_workflows_are_valid_yaml(self):
        for path in sorted((ROOT / '.github/workflows').glob('*.yml')):
            with self.subTest(path=path.name):
                self.assertIsInstance(yaml.safe_load(path.read_text()), dict)

    def test_pages_url_uses_current_repository_context(self):
        for name in ('links.yml', 'wheels.yml', 'verify_package_set.yml'):
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                self.assertIn('${{ github.repository_owner }}.github.io', workflow)
                self.assertIn('${{ github.event.repository.name }}', workflow)

    def test_feature_branch_forces_all_packages_and_verifies_final_set(self):
        reusable = self.workflow('wheels.yml')
        orchestrator = self.workflow('build_wheels.yml')
        verifier = self.workflow('verify_package_set.yml')

        self.assertIn("inputs.force-build || github.ref != 'refs/heads/main'", reusable)
        for package in ('mkl-service', 'numpy', 'scipy'):
            self.assertIn(f'name: {package}', orchestrator)
        self.assertIn("github.ref != 'refs/heads/main'", orchestrator)
        self.assertIn('uses: ./.github/workflows/verify_package_set.yml', orchestrator)
        self.assertIn('needs: [mkl-service, numpy, scipy]', orchestrator)
        self.assertIn('Expected one mkl-service wheel', verifier)
        self.assertIn('Expected one NumPy wheel', verifier)
        self.assertIn('Expected one SciPy wheel', verifier)
        self.assertIn('tools/verify_package_set.py', verifier)

    def test_single_package_force_workflows_keep_index_fallback(self):
        for name in ('force_mkl_service.yml', 'force_numpy.yml', 'force_scipy.yml'):
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                self.assertIn('force-build: true', workflow)
                self.assertNotIn('verify_package_set.yml', workflow)

        force_all = self.workflow('force_all.yml')
        self.assertEqual(force_all.count('force-build: true'), 3)
        self.assertIn('verify_package_set.yml', force_all)

    def test_tests_do_not_hot_copy_source_into_installed_package(self):
        workflow = self.workflow('wheels.yml')

        self.assertNotIn('helper=$(python tools/get_file_in_pkg _init_helper.py', workflow)
        self.assertIn('dependency-wheelhouse/*.whl', workflow)

    def test_numpy_runs_full_upstream_suite_after_ilp64_assertions(self):
        tests = (TOOLS / 'numpy_tests.py').read_text()

        self.assertIn("assert HAS_LAPACK64", tests)
        self.assertIn("label='full'", tests)
        self.assertIn("'--timeout=1800'", tests)
        self.assertNotIn("'-k'", tests)

    def test_nix_stays_empty_until_complete_fork_pins_exist(self):
        pins = (ROOT / 'nix/wheels.nix').read_text()
        updater = (TOOLS / 'update-nix-wheels').read_text()
        workflow = self.workflow('nix_flakes.yml')

        self.assertIn("'https://michael-denyer.github.io/numpy-mkl'", updater)
        self.assertIn(
            "'https://github.com/michael-denyer/numpy-mkl/releases/download/'",
            updater,
        )
        self.assertIn('did not resolve the complete fork package set', updater)
        self.assertIn('resolved repo package(s) outside this fork', updater)
        self.assertIn("github.repository == 'michael-denyer/numpy-mkl'", workflow)
        self.assertIn("github.event.workflow_run.conclusion == 'success'", workflow)
        self.assertIn("github.event.workflow_run.head_branch == 'main'", workflow)
        self.assertNotIn('Rebuild numpy', workflow)
        self.assertNotIn('Rebuild scipy', workflow)
        self.assertNotIn('Rebuild mkl-service', workflow)
        self.assertTrue(pins.rstrip().endswith('{}'))

    def test_operational_paths_do_not_point_to_upstream_fork(self):
        paths = (
            ROOT / '.github',
            ROOT / 'benchmarks',
            ROOT / 'tools',
            ROOT / 'templates',
        )
        matches = []
        for path in paths:
            for file in path.rglob('*'):
                if (
                    file.is_file()
                    and file.resolve() != Path(__file__).resolve()
                    and '__pycache__' not in file.parts
                    and 'urob' in file.read_text(errors='ignore')
                ):
                    matches.extend([file.relative_to(ROOT).as_posix()])

        self.assertEqual(matches, [])


class TestPackageSetIdentity(unittest.TestCase):
    def test_reads_distribution_identity_from_wheel_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            wheel = Path(temporary_directory) / 'mkl_service-2.8.0-cp312-none-any.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr(
                    'mkl_service-2.8.0.dist-info/METADATA',
                    'Metadata-Version: 2.1\nName: mkl-service\nVersion: 2.8.0\n',
                )

            identity = verify_package_set.WheelIdentity.read('mkl-service', wheel)

        self.assertEqual(identity.distribution, 'mkl-service')
        self.assertEqual(identity.version, '2.8.0')


class TestNixResolutionBoundary(unittest.TestCase):
    @staticmethod
    def package(name, url):
        return {
            'name': name,
            'version': '1.0',
            'wheels': [
                {
                    'url': url,
                    'hashes': {'sha256': '00' * 32},
                }
            ],
        }

    def fetch_wheels(self, packages):
        updater = runpy.run_path(str(TOOLS / 'update-nix-wheels'))
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        resolution = updater['Resolution'](
            '3.12', Path(temporary_directory.name), 'https://index.example'
        )

        def write_lock(args):
            lock = Path(args[-1])
            lock.write_text('lock')
            return ''

        resolution.run = write_lock
        with patch.object(updater['tomllib'], 'loads', return_value={'packages': packages}):
            return resolution.fetch_wheels(), updater['ResolutionError']

    def test_rejects_incomplete_fork_package_set(self):
        with self.assertRaisesRegex(Exception, 'complete fork package set'):
            self.fetch_wheels([])

    def test_rejects_repo_package_from_another_release_owner(self):
        fork = 'https://github.com/michael-denyer/numpy-mkl/releases/download/1/'
        packages = [
            self.package('mkl-service', fork + 'mkl_service.whl'),
            self.package('numpy', 'https://github.com/other/repo/releases/download/numpy.whl'),
            self.package('scipy', fork + 'scipy.whl'),
        ]

        with self.assertRaisesRegex(Exception, 'outside this fork'):
            self.fetch_wheels(packages)


if __name__ == '__main__':
    unittest.main()
