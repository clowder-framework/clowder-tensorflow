import os

def delete_temp_files(file_paths):
    for file_path in file_paths:
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print("Deleted file from disk", file_path)
        except Exception as e:
            print(e)