import pandas as pd

data = pd.read_csv("data/20250123_DeepOpenCRISPR-1_Dataset.tsv", sep='\t')

def compare_target(row):
    if pd.notna(row['Spacer']) and pd.notna(row['Target']):
        return 1 if row['Spacer'] == row['Target'][5:24] else 0
    return 0

data['ontarget'] = data.apply(compare_target, axis=1)

data.to_csv('data/20250123_DeepOpenCRISPR_final.csv')