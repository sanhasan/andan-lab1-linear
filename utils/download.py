import io
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download


def download_dataset(repo_id, filenames, output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    paths = [output_dir / filename for filename in filenames]
    if all(path.is_file() for path in paths):
        print("Датасет уже скачан.")
        return paths

    print("Скачиваем датасет...")
    for filename, path in zip(filenames, paths):
        if not path.is_file():
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=output_dir,
            )
    return paths

# перегоняем скаченные датасеты в размеченные данные
def load_images(parquet_paths, image_size):
    X, y = [], []
    for parquet_path in parquet_paths:
        rows = pq.read_table(parquet_path, columns=["imageA", "imageB"]).to_pylist()
        for row in rows:
            images = (
                (row["imageA"]["bytes"], -1),
                (row["imageB"]["bytes"], 1),
            )
            for image_bytes, label in images:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image = image.convert("RGB")
                    image = image.resize((image_size, image_size))
                    data = np.asarray(image, dtype=np.float32).ravel() / 255

                X.append(data)
                y.append(label)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8)
