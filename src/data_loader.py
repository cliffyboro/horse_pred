import pandas as pd
import numpy as np
from pathlib import Path
import time

class RaceDataLoader:
    def __init__(self, data_path="database/raceform.csv"):
        self.data_path = Path(data_path)
        self.uk_ire_courses = {
            'Aintree', 'Ascot', 'Ayr', 'Bangor-on-Dee', 'Bath', 'Beverley', 'Brighton',
            'Carlisle', 'Cartmel', 'Catterick', 'Chelmsford (AW)', 'Cheltenham', 
            'Chepstow', 'Chester', 'Doncaster', 'Epsom', 'Exeter', 'Ffos Las', 
            'Fontwell', 'Goodwood', 'Hamilton', 'Haydock', 'Hexham', 'Huntingdon',
            'Kelso', 'Kempton (AW)', 'Leicester', 'Lingfield (AW)', 'Ludlow', 
            'Market Rasen', 'Musselburgh', 'Newbury', 'Newcastle', 'Newmarket', 
            'Newton Abbot', 'Nottingham', 'Perth', 'Pontefract', 'Redcar', 'Ripon', 
            'Salisbury', 'Sandown', 'Sedgefield', 'Southwell', 'Southwell (AW)', 
            'Stratford', 'Taunton', 'Thirsk', 'Towcester', 'Uttoxeter', 'Warwick', 
            'Wetherby', 'Wincanton', 'Windsor', 'Wolverhampton (AW)', 'Worcester', 
            'Yarmouth', 'York', 'Newmarket (July)', 'Newmarket (Rowley)', 'Kempton', 
            'Lingfield', 'Plumpton',
            # Ireland
            'Ballinrobe (IRE)', 'Bellewstown (IRE)', 'Clonmel (IRE)', 'Cork (IRE)',
            'Curragh (IRE)', 'Down Royal (IRE)', 'Downpatrick (IRE)', 'Dundalk (AW) (IRE)',
            'Fairyhouse (IRE)', 'Galway (IRE)', 'Gowran (IRE)', 'Gowran Park (IRE)',
            'Kilbeggan (IRE)', 'Killarney (IRE)', 'Laytown (IRE)', 'Leopardstown (IRE)',
            'Limerick (IRE)', 'Listowel (IRE)', 'Naas (IRE)', 'Navan (IRE)', 
            'Punchestown (IRE)', 'Roscommon (IRE)', 'Sligo (IRE)', 'Tipperary (IRE)',
            'Tramore (IRE)', 'Wexford (IRE)', 'Thurles (IRE)'
        }
    
    def load_data(self, nrows=None, clean=True):
        print("Loading data...")
        start = time.time()
        
        df = pd.read_csv(self.data_path, nrows=nrows, low_memory=False)
        print(f"Raw rows loaded: {len(df):,}")
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['course'] = df['course'].astype(str).str.strip()
        
        # Filter
        df = df[df['course'].isin(self.uk_ire_courses)].copy()
        print(f"Rows after UK/Ire filter: {len(df):,}")
        
        if clean:
            df = self._basic_cleaning(df)
        
        print(f"Total time: {round(time.time()-start, 1)} seconds")
        return df
    
    def _basic_cleaning(self, df):
        print("Applying basic cleaning...")
        numeric_cols = ['age', 'ran', 'pos', 'draw', 'or', 'rpr', 'ts', 'prize']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        string_cols = ['horse', 'jockey', 'trainer', 'sex', 'going', 'type', 'class', 'race_name']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace('nan', np.nan)
        
        print("Basic cleaning completed.")
        return df


# Test
if __name__ == "__main__":
    loader = RaceDataLoader()
    df = loader.load_data(nrows=150000)
    print("\nShape:", df.shape)
    print("Date range:", df['date'].min(), "to", df['date'].max())