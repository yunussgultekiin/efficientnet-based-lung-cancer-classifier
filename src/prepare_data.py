from __future__ import annotations
import shutil
from collections import Counter
from pathlib import Path

from src.config import (
    CLASS_KEYWORDS,
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SPLITS,
)

class DataPreparer:
    def __init__(
        self,
        raw_dir: Path = RAW_DATA_DIR,
        processed_dir: Path = PROCESSED_DATA_DIR,
    ) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    def run(self) -> None:
        self._reset_processed_dir()
        self._copy_all_splits()
        self._print_summary()

    def is_ready(self) -> bool:
        return all(
            self._count_images(self.processed_dir / split / class_name) > 0
            for split in SPLITS
            for class_name in CLASS_NAMES
        )

    def ensure_ready(self) -> None:
        if not self.is_ready():
            self.run()

    def _reset_processed_dir(self) -> None:
        if self.processed_dir.exists():
            shutil.rmtree(self.processed_dir)

        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _copy_all_splits(self) -> None:
        for split in SPLITS:
            self._copy_split(split)

    def _copy_split(self, split: str) -> None:
        raw_split_dir = self.raw_dir / split
        processed_split_dir = self.processed_dir / split

        if not raw_split_dir.exists():
            raise FileNotFoundError(f"Split not found: {raw_split_dir}")

        processed_split_dir.mkdir(parents=True, exist_ok=True)

        for raw_class_dir in raw_split_dir.iterdir():
            if not raw_class_dir.is_dir():
                continue

            class_name = self._clean_class_name(raw_class_dir.name)
            target_class_dir = processed_split_dir / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            for image_path in self._get_image_files(raw_class_dir):
                target_path = target_class_dir / image_path.name
                shutil.copy2(image_path, target_path)

    def _clean_class_name(self, folder_name: str) -> str:
        for keyword, class_name in CLASS_KEYWORDS.items():
            if keyword in folder_name:
                return class_name

        raise ValueError(f"Unknown class folder: {folder_name}")

    def _get_image_files(self, folder: Path) -> list[Path]:
        return sorted(
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _count_images(self, folder: Path) -> int:
        if not folder.exists():
            return 0

        return len(self._get_image_files(folder))

    def _build_summary(self) -> dict[str, Counter[str]]:
        summary: dict[str, Counter[str]] = {}

        for split in SPLITS:
            split_counter: Counter[str] = Counter()

            for class_name in CLASS_NAMES:
                class_dir = self.processed_dir / split / class_name
                split_counter[class_name] = self._count_images(class_dir)

            summary[split] = split_counter

        return summary

    def _print_summary(self) -> None:
        summary = self._build_summary()
        dataset_total = 0

        print("\nProcessed Dataset Summary")
        print("-" * 30)

        for split, class_counts in summary.items():
            split_total = sum(class_counts.values())
            dataset_total += split_total

            print(f"\n{split}")
            for class_name, count in class_counts.items():
                print(f"{class_name}: {count}")
            print(f"total: {split_total}")

        print(f"\nDataset total: {dataset_total}")