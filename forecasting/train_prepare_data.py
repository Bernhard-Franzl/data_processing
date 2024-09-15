import numpy as np
import random
import torch

from _forecasting import prepare_data, load_data_dicts, prepare_data_lecture
from _dfguru import DataFrameGuru as DFG
from _forecasting import OccFeatureEngineer, LectureFeatureEngineer

dfg = DFG()
#for frequency in [ "1h", "5min","1min", "15min", "30min",]:
    
#    #### Control Randomness ####
#    np_rng = np.random.default_rng(seed=42)
#    torch_rng = torch.Generator()
#    torch_rng.manual_seed(42)
#    random.seed(42)

#    train_dict, val_dict, test_dict = prepare_data(
#        path_to_data_dir="data",
#        frequency=frequency,
#        feature_list=sorted(list(OccFeatureEngineer.permissible_features)),
#        dfguru = dfg,
#        rng = np_rng
#    )

#    # save data_dict
#    for room_id, df in train_dict.items():
#        dfg.save_to_csv(df, "data", f"room-{room_id}_freq-{frequency}_train_dict")
    
#    for room_id, df in val_dict.items():
#        dfg.save_to_csv(df, "data", f"room-{room_id}_freq-{frequency}_val_dict")
    
#    for room_id, df in test_dict.items():
#        dfg.save_to_csv(df, "data", f"room-{room_id}_freq-{frequency}_test_dict")




#### Control Randomness ####
np_rng = np.random.default_rng(seed=42)
torch_rng = torch.Generator()
torch_rng.manual_seed(42)
random.seed(42)

train_set, val_set, test_set = prepare_data_lecture(
    path_to_data_dir="data",
    feature_list=sorted(list(LectureFeatureEngineer.permissible_features)),
    dfguru = dfg,
    rng = np_rng
)

## save data_dict
#for room_id, df in train_dict.items():
#    dfg.save_to_csv(df, "data", f"lecture_train_dict")

#for room_id, df in val_dict.items():
#    dfg.save_to_csv(df, "data", f"lecture_val_dict")

#for room_id, df in test_dict.items():
#    dfg.save_to_csv(df, "data", f"lecture_test_dict")
