from pathlib import Path

from src.train_classifier import ClassificationTrainer


def main():

    dataset = Path("data") / "cleaned_data.csv"

    if not dataset.exists():

        print("\n❌ cleaned_data.csv not found.")
        print("Run Module 1 first.")

        return

    trainer = ClassificationTrainer(dataset)

    trainer.run()


if __name__ == "__main__":
    main()