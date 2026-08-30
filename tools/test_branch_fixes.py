#!/usr/bin/env python

import json
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
import package_set_plan  # noqa: E402
import verify_package_set  # noqa: E402
import write_build_info  # noqa: E402
from store_info import Build  # noqa: E402

RECIPE = f'sha256:{"a" * 64}'
OLD_RECIPE = f'sha256:{"b" * 64}'


class TestBuildIdentity(unittest.TestCase):
    def build(self, recipe=RECIPE, mkl='2026.1.0'):
        return Build(
            {
                'name': 'numpy',
                'version': '2.5.2',
                'python': 'cp312',
                'os': 'Linux',
                'mkl': mkl,
                'recipe': recipe,
            }
        )

    def test_matching_recipe_skips_build(self):
        store = {
            'numpy-2.5.2-cp312-linux': {
                'mkl': '2026.1.0',
                'recipe': RECIPE,
            }
        }

        self.assertTrue(self.build().exclude(store))

    def test_changed_recipe_rebuilds(self):
        store = {
            'numpy-2.5.2-cp312-linux': {
                'mkl': '2026.1.0',
                'recipe': OLD_RECIPE,
            }
        }

        self.assertFalse(self.build().exclude(store))

    def test_legacy_store_entry_rebuilds(self):
        store = {'numpy-2.5.2-cp312-linux': {'mkl': '2026.1.0'}}

        self.assertFalse(self.build().exclude(store))

    def test_missing_recipe_fails_closed(self):
        store = {'numpy-2.5.2-cp312-linux': {'mkl': '2026.1.0'}}

        self.assertFalse(self.build(recipe=None).exclude(store))

    def test_empty_and_malformed_recipes_fail_closed(self):
        store = {'numpy-2.5.2-cp312-linux': {'mkl': '2026.1.0', 'recipe': RECIPE}}

        self.assertFalse(self.build(recipe='').exclude(store))
        self.assertFalse(self.build(recipe='sha256:not-a-digest').exclude(store))

    def test_unknown_mkl_with_mkl_check_rebuilds(self):
        store = {'numpy-2.5.2-cp312-linux': {'mkl': '2026.1.0', 'recipe': RECIPE}}

        self.assertFalse(self.build(mkl=None).exclude(store, check_mkl=True))

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
            {'mkl': '2026.1.0', 'recipe': RECIPE},
        )

        with self.assertRaisesRegex(ValueError, 'missing its recipe digest'):
            self.build(recipe=None).merge_with(store)
        with self.assertRaisesRegex(ValueError, 'invalid recipe digest'):
            self.build(recipe='').merge_with(store)

    def test_digest_is_deterministic_and_package_specific(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recipe_files = {
                *build_recipe.COMMON_FILES,
                *(
                    path
                    for files in build_recipe.PACKAGE_FILES.values()
                    for path in files
                ),
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

    def test_build_information_writer_records_current_recipe(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / 'build.json'
            build = write_build_info.write_build_info(
                {
                    'name': 'numpy',
                    'version': '2.5.2',
                    'python': 'cp312',
                    'os': 'Linux',
                    'mkl': '2026.1.0',
                },
                output,
            )

            self.assertEqual(json.loads(output.read_text()), build)
            self.assertRegex(build['recipe'], r'^sha256:[0-9a-f]{64}$')


class TestWorkflowContracts(unittest.TestCase):
    def workflow(self, name):
        return (ROOT / '.github/workflows' / name).read_text()

    def test_workflows_are_valid_yaml(self):
        for path in sorted((ROOT / '.github/workflows').glob('*.yml')):
            with self.subTest(path=path.name):
                self.assertIsInstance(yaml.safe_load(path.read_text()), dict)

    def test_pages_url_uses_current_repository_context(self):
        for name in (
            'links.yml',
            'package_set.yml',
            'wheels.yml',
            'verify_package_set.yml',
        ):
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                self.assertIn('${{ github.repository_owner }}.github.io', workflow)
                self.assertIn('${{ github.event.repository.name }}', workflow)

    def test_publication_is_owned_after_exact_set_verification(self):
        orchestrator = self.workflow('build_wheels.yml')
        coordinator = self.workflow('package_set.yml')
        reusable = self.workflow('wheels.yml')
        verifier = self.workflow('verify_package_set.yml')

        self.assertIn('uses: ./.github/workflows/package_set.yml', orchestrator)
        self.assertIn('pull_request:', orchestrator)
        self.assertNotIn('\n  release:', reusable)
        self.assertIn('uses: ./.github/workflows/verify_package_set.yml', coordinator)
        release = coordinator.index('\n  release:')
        verify = coordinator.index('\n  verify-package-set:')
        self.assertGreater(release, verify)
        self.assertIn('needs: [plan, verify-package-set]', coordinator)
        self.assertIn("github.ref == 'refs/heads/main'", coordinator)
        self.assertIn('Expected one mkl-service wheel', verifier)
        self.assertIn('Expected one NumPy wheel', verifier)
        self.assertIn('Expected one SciPy wheel', verifier)
        self.assertIn('tools/verify_package_set.py', verifier)

    def test_every_force_workflow_routes_through_package_set(self):
        for name in ('force_mkl_service.yml', 'force_numpy.yml', 'force_scipy.yml'):
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                self.assertIn('uses: ./.github/workflows/package_set.yml', workflow)
                self.assertIn('force-build: true', workflow)

        force_all = self.workflow('force_all.yml')
        self.assertIn('uses: ./.github/workflows/package_set.yml', force_all)
        self.assertEqual(force_all.count('force-build: true'), 2)

    def test_package_set_callers_allow_nested_publication_permissions(self):
        for name in (
            'build_wheels.yml',
            'force_all.yml',
            'force_mkl_service.yml',
            'force_numpy.yml',
            'force_scipy.yml',
        ):
            workflow = yaml.safe_load(self.workflow(name))
            for job_name, job in workflow['jobs'].items():
                if job.get('uses') != './.github/workflows/package_set.yml':
                    continue
                with self.subTest(workflow=name, job=job_name):
                    self.assertEqual(job['permissions']['contents'], 'write')

        coordinator = yaml.safe_load(self.workflow('package_set.yml'))
        for job_name in (
            'preflight',
            'plan',
            'mkl-service',
            'numpy',
            'scipy',
            'verify-package-set',
        ):
            with self.subTest(coordinator_job=job_name):
                self.assertEqual(
                    coordinator['jobs'][job_name]['permissions']['contents'], 'read'
                )
        for job_name in ('links', 'release'):
            with self.subTest(coordinator_job=job_name):
                self.assertEqual(
                    coordinator['jobs'][job_name]['permissions']['contents'], 'write'
                )

    def test_dependency_artifacts_are_mandatory_and_no_published_repo_fallback(self):
        workflow = self.workflow('wheels.yml')

        self.assertNotIn('continue-on-error', workflow)
        self.assertNotIn('nullglob', workflow)
        self.assertIn('Expected one same-run mkl-service wheel', workflow)
        self.assertIn('Expected one same-run NumPy wheel', workflow)
        self.assertIn(
            'versions-${PACKAGE_NAME}-${PYTHON_TAG}-${RUNNER_OS}.json', workflow
        )

    def test_build_information_does_not_depend_on_runner_jq(self):
        workflow = self.workflow('wheels.yml')

        self.assertNotIn('jq -n', workflow)
        self.assertIn('tools/write_build_info.py', workflow)

    def test_tests_do_not_hot_copy_source_into_installed_package(self):
        workflow = self.workflow('wheels.yml')

        self.assertNotIn(
            'helper=$(python tools/get_file_in_pkg _init_helper.py', workflow
        )
        self.assertIn('dependency-wheelhouse/mkl-service', workflow)
        self.assertIn('dependency-wheelhouse/numpy', workflow)

    def test_numpy_runs_full_upstream_suite_after_ilp64_assertions(self):
        tests = (TOOLS / 'numpy_tests.py').read_text()

        self.assertNotIn('assert HAS_LAPACK64', tests)
        self.assertIn('HAS_LAPACK64 is expect_ilp64', tests)
        self.assertNotIn('add_dll_directory', tests)
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

    def test_regression_suite_is_wired_into_preflight(self):
        coordinator = self.workflow('package_set.yml')

        self.assertIn('tools/test_branch_fixes.py', coordinator)
        self.assertIn('ruff==0.14.6 format --check', coordinator)

    def test_skipped_links_cannot_skip_package_set_verification(self):
        coordinator = yaml.safe_load(self.workflow('package_set.yml'))

        self.assertEqual(
            coordinator['jobs']['plan']['outputs']['has_builds'],
            '${{ steps.plan.outputs.has_builds }}',
        )
        for job_name in (
            'mkl-service',
            'numpy',
            'scipy',
            'verify-package-set',
            'release',
        ):
            condition = coordinator['jobs'][job_name]['if']
            with self.subTest(job=job_name):
                self.assertIn('always()', condition)
                self.assertIn("needs.plan.result == 'success'", condition)
                self.assertIn("needs.plan.outputs.has_builds == 'true'", condition)

    def test_scipy_license_and_locked_pkgconf_are_preserved(self):
        workflow = self.workflow('wheels.yml')

        self.assertIn('cat LICENSES_bundled.txt >>LICENSE.txt', workflow)
        self.assertNotIn('choco install', workflow)
        self.assertIn('pkgconf.get_executable()', workflow)

    def test_recipe_includes_package_set_certification_inputs(self):
        required = {
            '.github/workflows/package_set.yml',
            '.github/workflows/verify_package_set.yml',
            'tools/package_set_plan.py',
            'tools/verify_package_set.py',
        }

        self.assertTrue(required.issubset(build_recipe.COMMON_FILES))

    def test_platform_policy_is_carried_by_matrix(self):
        config = yaml.safe_load((ROOT / 'ci-targets.yaml').read_text())

        for runner in config['defaults']['runners']:
            self.assertIsInstance(runner, dict)
            self.assertIn(runner['os'], {'Linux', 'Windows'})
            self.assertTrue(runner['numpy_ilp64'])
            self.assertIn('scipy_ilp64', runner)

    def test_platform_policy_rejects_unknown_os(self):
        fetch_matrix = runpy.run_path(str(TOOLS / 'fetch_matrix2'))
        config = yaml.safe_load((ROOT / 'ci-targets.yaml').read_text())
        config['defaults']['runners'][0]['os'] = 'Plan9'
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'targets.yaml'
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, 'Unknown runner policy OS'):
                fetch_matrix['Config']('numpy', path)

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


class TestPackageSetPlan(unittest.TestCase):
    @staticmethod
    def package(name, full_versions, stale_versions=()):
        def matrix(versions):
            include = [
                {
                    'runner': 'ubuntu-latest',
                    'python_version': version,
                    'container': 'manylinux',
                    'os': 'Linux',
                    'numpy_ilp64': True,
                    'scipy_ilp64': True,
                }
                for version in versions
            ]
            return {'include': include} if include else None

        return {
            'name': name,
            'version': '1.0',
            'tag': 'v1.0',
            'matrix': matrix(stale_versions),
            'full_matrix': matrix(full_versions),
        }

    def test_uses_common_coordinates_not_scipy_matrix(self):
        plan = package_set_plan.build_plan(
            {
                'mkl-service': self.package(
                    'mkl-service', ('cp311', 'cp312', 'cp313'), ('cp311',)
                ),
                'numpy': self.package('numpy', ('cp312', 'cp313'), ('cp312',)),
                'scipy': self.package('scipy', ('cp312', 'cp313'), ('cp313',)),
            }
        )

        self.assertEqual(
            [entry['python_version'] for entry in plan['matrix']['include']],
            ['cp312', 'cp313'],
        )

    def test_fully_cached_set_is_noop(self):
        packages = {
            name: self.package(name, ('cp312',))
            for name in ('mkl-service', 'numpy', 'scipy')
        }

        self.assertIsNone(package_set_plan.build_plan(packages)['matrix'])

    def test_force_rebuilds_full_common_set(self):
        packages = {
            name: self.package(name, ('cp312', 'cp313'))
            for name in ('mkl-service', 'numpy', 'scipy')
        }

        plan = package_set_plan.build_plan(packages, force=True)

        self.assertEqual(len(plan['matrix']['include']), 2)


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
        with patch.object(
            updater['tomllib'], 'loads', return_value={'packages': packages}
        ):
            return resolution.fetch_wheels(), updater['ResolutionError']

    def test_rejects_incomplete_fork_package_set(self):
        with self.assertRaisesRegex(Exception, 'complete fork package set'):
            self.fetch_wheels([])

    def test_rejects_repo_package_from_another_release_owner(self):
        fork = 'https://github.com/michael-denyer/numpy-mkl/releases/download/1/'
        packages = [
            self.package('mkl-service', fork + 'mkl_service.whl'),
            self.package(
                'numpy', 'https://github.com/other/repo/releases/download/numpy.whl'
            ),
            self.package('scipy', fork + 'scipy.whl'),
        ]

        with self.assertRaisesRegex(Exception, 'outside this fork'):
            self.fetch_wheels(packages)


if __name__ == '__main__':
    unittest.main()
