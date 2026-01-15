"""Data preprocessing module for job data"""

import re
from typing import Tuple

import pandas as pd

import config


class JobDataPreprocessor:
    """Preprocessor for job posting data"""

    def __init__(self, data_path: str = None):
        """
        Args:
            data_path: Path to Excel file
        """
        self.data_path = data_path or config.RAW_DATA_PATH
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load data from Excel file"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_excel(self.data_path, sheet_name="tcv")
        print(f"Loaded {len(self.df)} job postings")
        return self.df

    def normalize_salary(self, salary_str: str) -> Tuple[float, float]:
        """
        Normalize salary string to (min, max) in million VND

        Examples:
            '18 - 25 triệu' -> (18.0, 25.0)
            'Thoả thuận' -> (0.0, 0.0)
            'Tới 3,000 USD' -> (75.0, 75.0)  # Convert to VND
        """
        if pd.isna(salary_str) or salary_str == "Thoả thuận":
            return (0.0, 0.0)

        salary_str = str(salary_str).lower()

        # Handle USD - convert to VND (1 USD ~ 25000 VND = 25 triệu)
        if "usd" in salary_str:
            numbers = re.findall(r"[\d,]+", salary_str)
            if numbers:
                usd_amount = float(numbers[0].replace(",", ""))
                vnd_amount = usd_amount * 25  # Convert to million VND
                if "tới" in salary_str or "up to" in salary_str:
                    return (0.0, vnd_amount)
                return (vnd_amount, vnd_amount)

        # Extract numbers from string
        numbers = re.findall(r"\d+", salary_str)

        if not numbers:
            return (0.0, 0.0)

        numbers = [float(n) for n in numbers]

        if len(numbers) >= 2:
            # Range: "18 - 25 triệu"
            return (min(numbers), max(numbers))
        elif len(numbers) == 1:
            # Single value or "Tới X"
            if "tới" in salary_str or "trên" in salary_str:
                return (0.0, numbers[0])
            return (numbers[0], numbers[0])

        return (0.0, 0.0)

    def normalize_experience(self, exp_str: str) -> float:
        """
        Normalize experience string to years

        Examples:
            '3 năm' -> 3.0
            'Dưới 1 năm' -> 0.5
            'Không yêu cầu' -> 0.0
        """
        if pd.isna(exp_str):
            return 0.0

        exp_str = str(exp_str).lower()

        # No experience required
        if "không yêu cầu" in exp_str or "no experience" in exp_str:
            return 0.0

        # Less than 1 year
        if "dưới" in exp_str or "under" in exp_str:
            return 0.5

        # Extract numbers
        numbers = re.findall(r"\d+", exp_str)
        if numbers:
            return float(numbers[0])

        return 0.0

    def clean_location(self, location_str: str) -> str:
        """
        Clean and standardize location string

        Examples:
            'Hồ Chí Minh (mới)' -> 'Hồ Chí Minh'
            'Hồ Chí Minh (mới) & 9 nơi khác' -> 'Hồ Chí Minh'
        """
        if pd.isna(location_str):
            return "Unknown"

        location_str = str(location_str)

        # Remove (mới), (new), etc.
        location_str = re.sub(r"\s*\([^)]*\)", "", location_str)

        # Take first location if multiple
        if "&" in location_str:
            location_str = location_str.split("&")[0]

        return location_str.strip()

    def preprocess(self) -> pd.DataFrame:
        """
        Main preprocessing pipeline

        Returns:
            DataFrame with normalized features
        """
        if self.df is None:
            self.load_data()

        print("\nPreprocessing data...")

        # Create a copy
        df_processed = self.df.copy()

        # 1. Normalize Salary
        print("- Normalizing salary...")
        df_processed[["salary_min", "salary_max"]] = df_processed["Salary"].apply(
            lambda x: pd.Series(self.normalize_salary(x))
        )

        # 2. Normalize Experience
        print("- Normalizing experience...")
        df_processed["experience_years"] = df_processed["Experience"].apply(
            self.normalize_experience
        )

        # 3. Clean Location
        print("- Cleaning location...")
        df_processed["location_clean"] = df_processed["Job Address"].apply(
            self.clean_location
        )

        # 4. Handle missing values in text fields
        print("- Handling missing values...")
        text_columns = ["Title", "Job Requirements", "Job description", "benefit"]
        for col in text_columns:
            df_processed[col] = df_processed[col].fillna("")

        # 5. Create combined text for embedding
        print("- Creating combined text for embedding...")
        df_processed["combined_text"] = (
            df_processed["Title"]
            + " "
            + df_processed["Job Requirements"]
            + " "
            + df_processed["Job description"]
        )

        print("\nPreprocessing complete!")
        print(
            f"- Salary range: {df_processed['salary_min'].min():.1f} - {df_processed['salary_max'].max():.1f} million VND"
        )
        print(
            f"- Experience range: {df_processed['experience_years'].min():.1f} - {df_processed['experience_years'].max():.1f} years"
        )
        print(f"- Unique locations: {df_processed['location_clean'].nunique()}")
        print(f"- Unique companies: {df_processed['Name company'].nunique()}")

        return df_processed

    def save_processed_data(
        self, df: pd.DataFrame, filename: str = "jobs_processed.csv"
    ):
        """Save processed data to CSV"""
        output_path = f"{config.PROCESSED_DATA_PATH}{filename}"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved processed data to {output_path}")


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = JobDataPreprocessor()
    df_processed = preprocessor.preprocess()
    preprocessor.save_processed_data(df_processed)

    # Display sample
    print("\n" + "=" * 60)
    print("SAMPLE PROCESSED DATA:")
    print("=" * 60)
    print(
        df_processed[
            [
                "JobID",
                "Title",
                "salary_min",
                "salary_max",
                "experience_years",
                "location_clean",
            ]
        ].head(10)
    )
