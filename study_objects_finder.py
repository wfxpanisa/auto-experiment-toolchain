#!/usr/bin/python3

import subprocess

list_groups = ['music player', 'media player', 'video player', 'movie player']
list_libs = ['libgtk', 'libqt5core']

def run_shell_command(cmd):
    return subprocess.run(cmd, shell=True, universal_newlines=True, capture_output=True)


def get_list_packages(search_filter):
    lines = run_shell_command('apt search "{}"'.format(search_filter.replace(' ', '[ -]?'))).stdout.splitlines()
    list_packages = []
    for line in lines:
        if '...' in line:
            continue
        elif line.startswith(' '):
            continue
        elif line == '':
            continue
        elif line.startswith('lib'):
            continue
        elif 'plugin' in line:
            continue
        elif '-dev' in line:
            continue
        else:
                list_packages.append(line.split('/')[0])
    return list_packages


def check_if_depends(package_name, dependency_name):
    lines = run_shell_command('apt-cache depends {}'.format(package_name)).stdout.splitlines()
    return any(dependency_name in line for line in lines)


for filter_item in list_groups:
    for package in get_list_packages(filter_item):
        for lib in list_libs:
            if check_if_depends(package, lib):
                print('{};{};{}'.format(filter_item, package, lib))
