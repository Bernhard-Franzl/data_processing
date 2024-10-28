import numpy as np
import random
import torch

from _occupancy_forecasting import prepare_data, load_data_dicts, prepare_data_lecture, load_data_lecture
from _dfguru import DataFrameGuru as DFG
from _occupancy_forecasting import OccFeatureEngineer, LectureFeatureEngineer
dfg = DFG()


for frequency in ["1min", "5min", "15min", "30min", "1h"]:
    
    #### Control Randomness ####
    np_rng = np.random.default_rng(seed=42)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(42)
    random.seed(42)

    train_dict, val_dict, test_dict = prepare_data(
        path_to_data_dir="data",
        frequency=frequency,
        feature_list=sorted(list(OccFeatureEngineer.permissible_features)),
        dfguru = dfg,
        rng = np_rng
    )

    # save data_dict
    for room_id, df in train_dict.items():
        dfg.save_to_csv(df, f"data/freq_{frequency}", f"room-{room_id}_train_dict")
    
    for room_id, df in val_dict.items():
        dfg.save_to_csv(df, f"data/freq_{frequency}", f"room-{room_id}_val_dict")
    
    for room_id, df in test_dict.items():
        dfg.save_to_csv(df, f"data/freq_{frequency}", f"room-{room_id}_test_dict")

