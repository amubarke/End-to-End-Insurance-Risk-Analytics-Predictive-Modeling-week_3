import pandas as pd

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def delete_columns(self):
        cols_to_drop = ["NumberOfVehiclesInFleet"]
        self.df = self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns], errors="ignore")

    def convert_dtypes(self):
        # Columns to convert
        datetime_cols = ["TransactionMonth", "VehicleIntroDate"]
        float_cols = ["CapitalOutstanding","CalculatedPremiumPerTerm","TotalPremium",
                      "TotalClaims","SumInsured","CustomValueEstimate"]
        int_cols = ["TermFrequency"]
        
        categorical_cols = [
            "UnderwrittenCoverID","PolicyID","IsVATRegistered","Citizenship","LegalType",
            "Title","Language","Bank","AccountType","MaritalStatus","Gender","Country",
            "Province","PostalCode","MainCrestaZone","SubCrestaZone","ItemType","mmcode",
            "VehicleType","RegistrationYear","make","Model","Cylinders","bodytype",
            "AlarmImmobiliser","TrackingDevice","NewVehicle","WrittenOff","Rebuilt",
            "Converted","CrossBorder","CoverCategory","CoverType","CoverGroup","Section",
            "Product","StatutoryClass","StatutoryRiskType"
        ]

        # Convert datetimes
        for col in datetime_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        # Convert floats
        for col in float_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Convert ints
        for col in int_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype("Int64")

        # Convert to category
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

    def fill_missing(self):
        # Fill categorical missing → "Unknown"
        cat_cols = self.df.select_dtypes(include=["category"]).columns
        for col in cat_cols:
            self.df[col] = self.df[col].cat.add_categories("Unknown").fillna("Unknown")

        # Fill numeric missing → 0
        num_cols = self.df.select_dtypes(include=["float64", "int64", "Int64"]).columns
        for col in num_cols:
            self.df[col] = self.df[col].fillna(0)

        # Date columns fill with earliest date
        date_cols = self.df.select_dtypes(include=["datetime64[ns]"]).columns
        for col in date_cols:
            self.df[col] = self.df[col].fillna(self.df[col].min())

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()

    def get_clean_data(self):
        return self.df

    def full_clean(self):
        self.delete_columns()
        self.convert_dtypes()
        self.fill_missing()
        self.remove_duplicates()
        return self.df
