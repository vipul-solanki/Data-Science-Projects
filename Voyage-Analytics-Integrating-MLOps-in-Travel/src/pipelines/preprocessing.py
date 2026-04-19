import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, data_path="./travel_capstone"):
        self.data_path = data_path

    def load_data(self):
        self.users = pd.read_csv(f"{self.data_path}/users.csv")
        self.flights = pd.read_csv(f"{self.data_path}/flights.csv")
        self.hotels = pd.read_csv(f"{self.data_path}/hotels.csv")

    def merge_data(self):
        df = pd.merge(self.flights, self.users, left_on="userCode", right_on="code", how="left")
        df = pd.merge(df, self.hotels, on=["travelCode", "userCode"], how="left")
        self.df = df

    def clean_data(self):
        self.df.drop_duplicates(inplace=True)
        self.df.ffill(inplace=True)

    def feature_engineering(self):

        hotel_map = self.hotels[["travelCode", "userCode", "name"]].rename(
            columns={"name": "hotel_name_original"}
        )


        self.df["hotel_name_original"] = self.df["name_y"]
        self.df["place_original"] = self.df["place"]


        self.df = pd.merge(
            self.df,
            hotel_map,
            on=["travelCode", "userCode"],
            how="left"
        )


        if "age" in self.df.columns:
            self.df["age_group"] = pd.cut(
                self.df["age"],
                bins=[0, 18, 30, 50, 100],
                labels=["Teen", "Young", "Adult", "Senior"]
            )

            self.df["age_group"] = self.df["age_group"].astype(str)

            le = LabelEncoder()
            self.df["age_group"] = le.fit_transform(self.df["age_group"])

        # Encode categorical columns EXCEPT hotel_name_original
        le = LabelEncoder()

        for col in self.df.select_dtypes(include="object").columns:
            # 🚨 SKIP columns we want to preserve
            if col in ["hotel_name_original"]:
                continue

            self.df[col] = le.fit_transform(self.df[col].astype(str))

    def get_data(self):
        return self.df