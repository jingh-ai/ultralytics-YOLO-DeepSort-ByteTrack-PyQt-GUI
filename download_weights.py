import os
import requests
from zipfile import ZipFile


def download_dropbox_folder(dropbox_link, output_folder):
    """
    Downloads a folder from a Dropbox shared link.
    
    Args:
        dropbox_link (str): The Dropbox shared link (ensure dl=1 for direct download).
        output_folder (str): The folder where the downloaded content will be extracted.
    """
    # Step 1: Prepare download link
    if "dl=0" in dropbox_link:
        dropbox_link = dropbox_link.replace("dl=0", "dl=1")
    elif "?dl=0" not in dropbox_link and "?dl=1" not in dropbox_link:
        dropbox_link += "?dl=1"

    # Step 2: Download the file
    zip_filename = os.path.join(output_folder, "temp_download.zip")
    print(f"Downloading from {dropbox_link}...")
    try:
        with requests.get(dropbox_link, stream=True) as response:
            response.raise_for_status()
            with open(zip_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded to {zip_filename}")
    except Exception as e:
        print(f"Error during download: {e}")
        return

    # Step 3: Extract the ZIP file
    try:
        with ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        print(f"Extracted files to {output_folder}")
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
    finally:
        # Clean up temporary ZIP file
        os.remove(zip_filename)

    # Step 4: Verify and organize extracted files
    if os.path.exists(output_folder):
        print(f"Folder downloaded and extracted successfully to {output_folder}")
    else:
        print(f"Failed to extract files to {output_folder}")

if __name__ == "__main__":
    DETECTOR_URL = "https://www.dropbox.com/scl/fo/316wnrqqbir6da06629c6/AO6qqum3OkNtQN-d-ElL3w0?rlkey=63o6yuxkl63gsr72ldthf4utb&st=soq12k67&dl=0"
    DETECTOR_DIR = "weights/detection"
    if not os.path.exists(DETECTOR_DIR):
        os.makedirs(DETECTOR_DIR)
    download_dropbox_folder(DETECTOR_URL, DETECTOR_DIR)

    POSE_URL = "https://www.dropbox.com/scl/fo/96npbz8pcl3kuq0n5lkh8/AGFYwXjXravFI1NASMV9vAQ?rlkey=3a7ov55thf7w2j0ls1odqs01o&st=10k3no41&dl=0"
    POSE_DIR = "weights/pose"
    if not os.path.exists(POSE_DIR):
        os.makedirs(POSE_DIR)
    download_dropbox_folder(POSE_URL, POSE_DIR)

    SEG_URL = "https://www.dropbox.com/scl/fo/akxdb70nwkft1yjl7qmbt/AKyoi4egqjG6hb1fhkhKbPA?rlkey=xuw21bqzpy3vwb7z940rfukvj&st=9qnsz7oq&dl=0"
    SEG_DIR = "weights/segmentation"
    if not os.path.exists(SEG_DIR):
        os.makedirs(SEG_DIR)
    download_dropbox_folder(SEG_URL, SEG_DIR)
