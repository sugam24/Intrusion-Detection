import os
import kagglehub
import zipfile
import shutil

def setup_dataset():
    # 1. Define paths
    project_dir = os.getcwd()  # Our project folder (where venv exists)
    dataset_dir = os.path.join(project_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)  # Create if doesn't exist

    # 2. Download dataset (replace with your dataset ID)
    print("Downloading dataset...")
    download_path = kagglehub.dataset_download("hassan06/nslkdd")
    print(f"Downloaded to temporary location: {download_path}")

    # 3. Move all files to dataset folder
    for item in os.listdir(download_path):
        src = os.path.join(download_path, item)
        dst = os.path.join(dataset_dir, item)
        
        # Handle both files and subdirectories
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.move(src, dst)

    # 4. Check for zip files and extract
    for file in os.listdir(dataset_dir):
        if file.endswith('.zip'):
            zip_path = os.path.join(dataset_dir, file)
            print(f"Extracting {file}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            os.remove(zip_path)  # Delete the zip after extraction

    print(f"Dataset ready at: {dataset_dir}")
    print(f"Files: {os.listdir(dataset_dir)}")

if __name__ == "__main__":
    setup_dataset()