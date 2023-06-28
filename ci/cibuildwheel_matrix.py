import os
import subprocess
import json

platforms = ['linux', 'macos', 'windows']
runners = {
    'linux': 'ubuntu-22.04',
    'macos': 'macos-11',
    'windows': 'windows-2022',
}

targets = []
for platform in platforms:

    platform_targets = subprocess.check_output(['cibuildwheel', '--platform', platform, '--print-build-identifiers'])
    platform_targets = platform_targets.decode('utf-8')
    platform_targets = platform_targets.strip('\n').split('\n')

    for platform_target in platform_targets:

        targets.append({
            'os': runners[platform],
            'build': platform_target,
        })

with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
    print('matrix=' + json.dumps({'target': targets}), file=fh)
