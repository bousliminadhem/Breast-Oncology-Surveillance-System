import pathlib

def generate_tree(path, indent=""):
    path = pathlib.Path(path)
    
    # Get a sorted list of files and directories, ignoring hidden ones (starting with .)
    items = sorted([item for item in path.iterdir() if not item.name.startswith('.')])
    
    for i, item in enumerate(items):
        # Check if this is the last item in the current folder to use the "elbow"
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        print(f"{indent}{connector}{item.name}")
        
        # If it's a directory, recurse into it
        if item.is_dir():
            # If it was the last item, we don't draw a vertical line for the next level
            new_indent = indent + ("    " if is_last else "│   ")
            generate_tree(item, new_indent)

if __name__ == "__main__":
    # Change '.' to the specific path you want to scan
    root_dir = "" 
    
    print(f"{pathlib.Path(root_dir).resolve().name}/")
    generate_tree(root_dir)