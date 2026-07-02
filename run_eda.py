from pathlib import Path
from src.eda import EDA


def main():

    dataset = Path("data") / "cleaned_data.csv"

    if not dataset.exists():

        print("❌ cleaned_data.csv not found.")
        print("Run Module 1 first.")

        return

    eda = EDA(dataset)

    eda.run()


if __name__ == "__main__":
    main()