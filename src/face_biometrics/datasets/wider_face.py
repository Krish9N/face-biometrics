from pathlib import Path

from .base import FaceDataset
from .types import BoundingBox, FaceAnnotation, FaceSample


class WiderFaceDataset(FaceDataset):

    def __init__(
        self,
        image_root: str | Path,
        annotation_file: str | Path,
    ):
        self.image_root = Path(image_root)
        self.annotation_file = Path(annotation_file)

        self.samples = self._load_annotations()

    def _load_annotations(self) -> list[FaceSample]:
        samples = []

        with self.annotation_file.open("r") as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0

        while i < len(lines):

            # Image path
            image_path = lines[i]
            i += 1

            # Number of annotated faces
            num_faces = int(lines[i])
            i += 1

            faces = []

            for _ in range(num_faces):
                parts = lines[i].split()
                i += 1

                if len(parts) < 4:
                    raise ValueError(
                        f"Invalid annotation line: {lines[i - 1]}"
                    )

                x = float(parts[0])
                y = float(parts[1])
                width = float(parts[2])
                height = float(parts[3])

                # WIDER FACE may contain invalid annotations.
                invalid = int(parts[7]) if len(parts) > 7 else 0

                if invalid:
                    continue

                bbox = BoundingBox(
                    x1=x,
                    y1=y,
                    x2=x + width,
                    y2=y + height,
                )

                faces.append(
                    FaceAnnotation(
                        bbox=bbox,
                        landmarks=None,
                        identity=None,
                    )
                )

            samples.append(
                FaceSample(
                    image_path=str(self.image_root / image_path),
                    faces=faces,
                )
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> FaceSample:
        return self.samples[index]