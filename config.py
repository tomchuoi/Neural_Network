import importlib

# List of modules to import
modules_to_import = ['pickle', 'copy', 'os', 'shutil', 'cv2', 'tkinter', 'PIL', 'numpy']

def import_module(module_name):
    try:
        # Try importing the module
        return importlib.import_module(module_name)
    except ImportError:
        # If the module is not found, import it dynamically
        globals()[module_name] = importlib.import_module(module_name)
        return globals()[module_name]

# Import the modules from the list
for module_name in modules_to_import:
    if module_name not in globals():
        imported_module = import_module(module_name)
        print(f'{module_name} imported successfully.')


# If any of the modules in the 'modules_to_import' list are not imported,
# they will be imported dynamically and the success message will be printed.
