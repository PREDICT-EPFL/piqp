import os
import subprocess
import json

platforms = [
    ('linux', 'ubuntu-22.04', 'x86_64,i686'),
    ('linux', 'ubuntu-22.04-arm', 'aarch64'),
    ('macos', 'macos-13', 'x86_64'),
    ('macos', 'macos-14', 'arm64'),
    ('windows', 'windows-2022', 'AMD64'),
#    ('windows', 'windows-11-arm', 'ARM64')
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
        arch = platform_target.split('_', 1)[1]
        if arch not in grouped_targets:
            grouped_targets[arch] = []
        grouped_targets[arch].append(platform_target)

    for arch, group_targets in grouped_targets.items():
        targets.append({
            'arch': arch,
            'os': runner,
            'build': ' '.join(group_targets),
        })

print(targets)

with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
    print('matrix=' + json.dumps({'target': targets}), file=fh)
