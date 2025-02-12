import os
import shutil
import argparse

def copy_folder_contents(src_folder, dst_folder):
    """
    Copies all contents from src_folder to dst_folder.
    If dst_folder does not exist, it will be created.
    """
    try:
        # Check if the source folder exists
        if not os.path.exists(src_folder):
            print(f"Source folder '{src_folder}' does not exist.")
            return

        # Create the destination folder if it doesn't exist
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
            
        max_dst_id = max(map(int, filter(lambda s : s.isdigit(), os.listdir(dst_folder) + ["0"])))

        # Iterate over all the items in the source folder
        for item in os.listdir(src_folder):
            src_item = os.path.join(src_folder, item)
            dst_item = os.path.join(dst_folder, item)

            # If the item is a file, copy it
            if os.path.isfile(src_item):
                shutil.copy2(src_item, dst_item)
            
            # If the item is a directory, recursively copy it
            elif os.path.isdir(src_item):
                # Rename if digit
                if  max_dst_id > 0 and item.isdigit():
                    new_dst_id = max_dst_id + int(item)
                    dst_item = os.path.join(dst_folder, str(new_dst_id))
                shutil.copytree(src_item, dst_item)

        print(f"{len(os.listdir(src_folder))} items copied from {src_folder} to {dst_folder} ({len(os.listdir(dst_folder))} items in total).")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Copy contents of one folder to another.")
    parser.add_argument("src_folder", help="The source folder to copy from.")
    parser.add_argument("dst_folder", help="The destination folder to copy to.")

    args = parser.parse_args()

    copy_folder_contents(args.src_folder, args.dst_folder)