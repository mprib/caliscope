import os

def find_modules(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        if '.venv' in dirs:
            dirs.remove('.venv') # Remove the .venv directory from the search
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                print(os.path.join(subdir, file))

if __name__ == '__main__':
    root_dir = '.' # Replace with the root directory you want to search
    find_modules(root_dir)
