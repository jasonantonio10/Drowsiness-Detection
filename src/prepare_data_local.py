from pathlib import Path
import shutil
from tqdm import tqdm

def main():
    root_dir = Path(__file__).resolve().parents[1]
    print(root_dir)
    
    data_dir = root_dir / "data"
    print(data_dir)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    (train_dir / "opened").mkdir(parents=True, exist_ok=True)
    (train_dir / "closed").mkdir(parents=True, exist_ok=True)
    (val_dir / "opened").mkdir(parents=True, exist_ok=True)
    (val_dir / "closed").mkdir(parents=True, exist_ok=True)

    def copy_files(src: Path, dest: Path):
        """
        Copy all image files recursively from source folder (src) to destination folder (dest)
        """
        exts = {".jpg", "jpeg", ".png"}
        files = [f for f in src.rglob("*") if f.suffix.lower() in exts]
        dest.mkdir(parents=True, exist_ok=True)
        
        count = 0

        for f in tqdm(files, desc=f"Copying {src.name}", unit="file"):
            shutil.copy2(f, dest / f.name)
            count += 1
        
        return count
    
    # Copy train/test images
    train_closed = copy_files(data_dir / "TrainingSet" / "TrainingSet" / "Closed", train_dir / "closed")
    train_opened = copy_files(data_dir / "TrainingSet" / "TrainingSet" / "Opened", train_dir / "opened")
    val_closed = copy_files(data_dir / "TestSet" / "TestSet" / "Closed", val_dir / "closed")
    val_opened = copy_files(data_dir / "TestSet" / "TestSet" / "Opened", val_dir / "opened")
    imp_closed = copy_files(data_dir / "ImprovementSet" / "ImprovementSet" / "ImprovementSet" / "Closed", train_dir / "closed")
    imp_opened = copy_files(data_dir / "ImprovementSet" / "ImprovementSet" / "ImprovementSet" / "Opened", train_dir / "opened")

    print("Copied:")
    print(f" Train: {train_opened+imp_opened} opened, {train_closed+imp_closed} closed")
    print(f" Val:   {val_opened} opened, {val_closed} closed")




if __name__ == "__main__":
    main()