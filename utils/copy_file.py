from shutil import copy
import os

def copy_file(source_file: str,
              destination_folder: str,
              new_file_name: str
              ):
    if not os.path.isfile(source_file):
        print(f"Error: The source file '{source_file}' does not exist.")
        return
    if not os.path.isdir(destination_folder):
        print(f"Error: The destination folder '{destination_folder}' does not exist.")
        return
    
    try:
        destination_file = os.path.join(destination_folder, new_file_name)
        copy(source_file, destination_file)
    except Exception as e:
        print(f"Error occurred while copying the file: {e}")