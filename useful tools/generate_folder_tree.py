import pathlib

def generate_tree(path, file, indent=""):
    path = pathlib.Path(path)
    
    # FILTER: Only include items if they are directories AND not hidden
    items = sorted([item for item in path.iterdir() if item.is_dir() and not item.name.startswith('.')])
    
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        file.write(f"{indent}{connector}{item.name}/\n")
        
        # Since we filtered for directories above, we can call this directly
        new_indent = indent + ("    " if is_last else "│   ")
        generate_tree(item, file, new_indent)

if __name__ == "__main__":
    root_dir = r"D:\Projet Fédérateur\AI\Breast Oncology Surveillance System"
    output_filename = "folder_structure.txt" # Changed extension to .txt for clarity
    
    path_obj = pathlib.Path(root_dir).resolve()
    
    if path_obj.exists():
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"{path_obj.name}/\n")
            generate_tree(path_obj, f)
        print(f"Folder structure has been successfully saved to {output_filename}")
    else:
        print("The specified path does not exist.")