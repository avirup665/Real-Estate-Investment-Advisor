from pathlib import Path

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineering


def print_summary(df):

    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumns\n")

    for column in df.columns:
        print(f"✓ {column}")

    print("\nMissing Values\n")

    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("No missing values found.")
    else:
        print(missing)

    print("\nData Types\n")
    print(df.dtypes)

    print("\nFirst Five Rows\n")
    print(df.head())


def verify_columns(df):

    required_columns = [

        "Price_per_SqFt",

        "Future_Price_5Y",

        "Good_Investment",

        "Investment_Score"

    ]

    print("\n" + "=" * 60)
    print("VERIFYING REQUIRED COLUMNS")
    print("=" * 60)

    missing = []

    for col in required_columns:

        if col in df.columns:

            print(f"✓ {col}")

        else:

            print(f"✗ {col}")

            missing.append(col)

    if len(missing) > 0:

        raise Exception(
            f"\nMissing Columns : {missing}"
        )


def main():

    print("=" * 60)
    print("REAL ESTATE INVESTMENT ADVISOR")
    print("MODULE 1")
    print("=" * 60)

    dataset = Path("data") / "india_housing_prices.csv"

    if not dataset.exists():

        print("\nDataset not found!")

        print(dataset.resolve())

        return

    try:

        preprocessor = DataPreprocessor(dataset)

        df = preprocessor.preprocess()

        engineer = FeatureEngineering(df)

        df = engineer.run()

        output = Path("data") / "cleaned_data.csv"

        df.to_csv(

            output,

            index=False

        )

        verify_columns(df)

        print_summary(df)

        print("\n" + "=" * 60)
        print("MODULE 1 COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print("\nDataset saved at\n")

        print(output.resolve())

    except Exception as e:

        print("\nERROR OCCURRED\n")

        print(e)


if __name__ == "__main__":

    main()