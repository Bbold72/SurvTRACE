import pandas as pd
from pathlib import Path
import pickle
import os


for file_name in os.listdir(Path('results')):

    with open(Path('results', file_name), 'rb') as f:
        result = pickle.load(f)
        
    result = pd.DataFrame(result)

    print(file_name)
    print(result.agg(['mean', 'std']).transpose())