import os
import subprocess
import json

platforms = ['linux', 'macos', 'windows']
runners = {
    'linux': 'ubuntu-22.04',
    'macos': 'macos-12',
    'windows': 'windows-2022',
}

targets = []
for platform in platforms:

    platform_targets = subprocess.check_output(['cibuildwheel', '--platform', platform, '--print-build-identifiers'])
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
            'os': runners[platform],
            'build': ' '.join(group_targets),
        })

# print('matrix=' + json.dumps({'target': targets}))

with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
    print('matrix=' + json.dumps({'target': targets}), file=fh)
