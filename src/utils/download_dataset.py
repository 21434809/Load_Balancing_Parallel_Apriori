import kagglehub
import os
import shutil

# Create data directory if it doesn't exist
data_dir = "../data"
os.makedirs(data_dir, exist_ok=True)

# Download latest version (to default location)
path = kagglehub.dataset_download("psparks/instacart-market-basket-analysis")
print("Downloaded to:", path)

# Move files to data directory
if os.path.exists(path):
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(data_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"Files moved to: {data_dir}")