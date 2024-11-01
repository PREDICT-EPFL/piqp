import os
import subprocess
import json

platforms = [
    ('linux', 'ubuntu-22.04', 'x86_64,i686,aarch64'),
    ('macos', 'macos-12', 'x86_64'),
    ('macos', 'macos-14', 'arm64'),
    ('windows', 'windows-2022', 'x86,AMD64')
]

targets = []
for platform, runner, archs in platforms:

    platform_targets = subprocess.check_output(['cibuildwheel', '--platform', platform, '--archs', archs,
                                                '--print-build-identifiers'])
    platform_targets = platform_targets.decode('utf-8')
    platform_targets = platform_targets.strip('\n').split('\n')

    # group targets by python version
    grouped_targets = {}
    for platform_target in platform_targets:
        python_version = platform_target.split('-')[0]
        if python_version not in grouped_targets:
            grouped_targets[python_version] = []
        grouped_targets[python_version].append(platform_target)

    for python_version, group_targets in grouped_targets.items():
        targets.append({
            'version': python_version,
            'os': runner,
            'build': ' '.join(group_targets),
        })

print(targets)

with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
    print('matrix=' + json.dumps({'target': targets}), file=fh)
