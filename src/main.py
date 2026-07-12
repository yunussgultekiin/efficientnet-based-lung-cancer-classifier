from __future__ import annotations
import argparse
from pathlib import Path
from src.config import REPORT_DIR
from src.evaluate import EvaluationPipeline
from src.predict import BatchPredictor, Predictor
from src.prepare_data import DataPreparer
from src.train import TrainingPipeline

class Pipeline:
    def prepare_data(self) -> None:
        DataPreparer().run()

    def train(self) -> None:
        DataPreparer().ensure_ready()
        TrainingPipeline().run()

    def evaluate(self) -> None:
        EvaluationPipeline().run()

    def predict(self, path: Path) -> None:
        if path.is_dir():
            self._predict_folder(path)
        else:
            self._predict_image(path)

    def _predict_image(self, image_path: Path) -> None:
        prediction = Predictor().predict(image_path)
        print(f"class: {prediction.class_name}")
        print(f"confidence: {prediction.confidence:.4f}")

    def _predict_folder(self, folder: Path) -> None:
        batch_predictor = BatchPredictor()
        predictions = batch_predictor.predict_folder(folder)
        batch_predictor.save_results(predictions, REPORT_DIR / "predictions.csv")

        for prediction in predictions:
            print(f"{prediction.image_path.name}: {prediction.class_name} ({prediction.confidence:.4f})")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EfficientNet-B0 lung cancer classification pipeline"
    )

    parser.add_argument(
        "command",
        choices=["prepare-data", "train", "evaluate", "predict"],
    )
    parser.add_argument("--path", type=Path, default=None)

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    pipeline = Pipeline()

    if args.command == "prepare-data":
        pipeline.prepare_data()
    elif args.command == "train":
        pipeline.train()
    elif args.command == "evaluate":
        pipeline.evaluate()
    elif args.command == "predict":
        if args.path is None:
            raise ValueError("--path is required for the predict command")
        pipeline.predict(args.path) 

if __name__ == "__main__":
    main()