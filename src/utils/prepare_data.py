import json

import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config


def load_data(config: Config):
    with open(config.paths.path_to_train_json, 'r') as f:
        train_js = json.load(f)

    train_df = pd.DataFrame(train_js['annotations'])
    train_df.image_id = train_df.image_id + '.jpg'
    
    train_ann, valid_ann = train_test_split(
        train_df,
        random_state=config.seed,
        test_size=config.valid_size
    )
    return train_ann, valid_ann
