#!/usr/bin/env python
import os
import sys

base_dir = os.getcwd()

# List of directories to create
directories = [
    'src/vision_s8/api/routes',
    'src/vision_s8/core',
    'src/vision_s8/services',
    'src/vision_s8/models',
    'src/vision_s8/utils',
    'src/vision_s8/data/prompts',
    'tests',
    'uploads',
    'outputs'
]

# Create all directories
print('Creating directories using os.makedirs()...\n')
for dir_path in directories:
    os.makedirs(dir_path, exist_ok=True)
    full_path = os.path.join(base_dir, dir_path)
    print(f'✓ Created: {dir_path}')

print('\n' + '='*70)
print('DIRECTORY STRUCTURE')
print('='*70 + '\n')

# Display tree structure
def print_tree(path, prefix="", is_last=True):
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return
    
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    
    for i, dir_name in enumerate(dirs):
        is_last_entry = (i == len(dirs) - 1)
        current_prefix = "└── " if is_last_entry else "├── "
        print(prefix + current_prefix + dir_name + "/")
        
        next_prefix = prefix + ("    " if is_last_entry else "│   ")
        print_tree(os.path.join(path, dir_name), next_prefix, is_last_entry)

print(f"{os.path.basename(base_dir)}/")
print_tree(base_dir)

print('\n' + '='*70)
print('All directories created successfully!')
print('='*70)
