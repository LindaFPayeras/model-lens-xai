import random
import pandas as pd
from pathlib import Path
from PIL import Image


# Obtener la raíz del proyecto independientemente
# del directorio desde el que se ejecute el notebook.
ROOT = Path(__file__).resolve().parents[2]

# Ruta del dataset.
DATASET_PATH = ROOT / "data" / "imagenette2-320"


def extractData(
    dir_dataset: str | Path = DATASET_PATH / "val",
    number_class: int = 30,
    seed: int = 44
):
    dir_dataset = Path(dir_dataset)

    if not dir_dataset.is_dir():
        raise FileNotFoundError(
            f"No existe el directorio del dataset: {dir_dataset}"
        )

    rng = random.Random(seed)
    selected_img = []

    for d in sorted(dir_dataset.iterdir()):
        if not d.is_dir():
            continue

        images = sorted(d.glob("*.JPEG"))

        if len(images) < number_class:
            raise ValueError(
                f"No hay suficientes imágenes en {d.name}"
            )

        sample = rng.sample(images, number_class)

        for image in sample:
            selected_img.append({
                "image_id": image.stem,
                "path": str(image.relative_to(ROOT)),
                "true_label": d.name
            })

    return pd.DataFrame(selected_img)


def get_img(row):

    # Reconstruir la ruta absoluta a partir de la relativa.
    image_path = ROOT / row["path"]

    if not image_path.is_file():
        raise FileNotFoundError(
            f"No existe la imagen: {image_path}"
        )

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    return {
        "image_id": row["image_id"],
        "path": row["path"],
        "true_label": row["true_label"],
        "image": image
    }

def createFil_selectedImages(dataframe: pd.DataFrame):
    dataframe.to_csv(ROOT / "metadata" / "selected_images.csv")

    return dataframe.head(10)
