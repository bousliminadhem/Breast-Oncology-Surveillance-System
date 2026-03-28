import pathlib

def generate_tree(path, file, indent=""):
    path = pathlib.Path(path)
    
    # Get a sorted list of files and directories, ignoring hidden ones
    items = sorted([item for item in path.iterdir() if not item.name.startswith('.')])
    
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        # Write the line to the file instead of printing to console
        file.write(f"{indent}{connector}{item.name}\n")
        
        if item.is_dir():
            new_indent = indent + ("    " if is_last else "│   ")
            generate_tree(item, file, new_indent)

if __name__ == "__main__":
    # Your specific path
    root_dir = r"D:\Projet Fédérateur\AI\Breast Oncology Surveillance System\data"
    output_filename = "tree.py"
    
    path_obj = pathlib.Path(root_dir).resolve()
    
    with open(output_filename, "w", encoding="utf-8") as f:
        # Write the root directory name first
        f.write(f"{path_obj.name}/\n")
        # Generate the rest of the tree into the file
        generate_tree(root_dir, f)
        
    print(f"Tree structure has been successfully saved to {output_filename}")