import os
def create_folder_if_needed(path, folder_name):
    full_path = os.path.join(path, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path