import numpy as np
import random
import torch

from _forecasting import prepare_data, load_data_dicts, prepare_data_lecture, load_data_lecture
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




################## Lecture Data ##################
np_rng = np.random.default_rng(seed=42)
torch_rng = torch.Generator()
torch_rng.manual_seed(42)
random.seed(42)

train_set, val_set, test_set = prepare_data_lecture(
    path_to_data_dir="data",
    feature_list=sorted(list(LectureFeatureEngineer.permissible_features)),
    dfguru = dfg,
    rng = np_rng,
    split_by="time"
)

dfg.save_to_csv(train_set, "data", f"lecture_train_set")
dfg.save_to_csv(val_set, "data", f"lecture_val_set")
dfg.save_to_csv(test_set, "data", f"lecture_test_set")

train_df, val_df, test_df = load_data_lecture("data", dfguru=dfg)

#LVA typen ausdünnen
# VL: VL, VO, KO
# UE: UE, AG IK, PR, PS
# KS: KS, VU, KV, RE, UV
# SE: SE
#print(train_df.columns)

## value counts of all the columns to spot class imbalance
#for col in train_df.columns:
#    print(f"Column: {col}")
#    print(train_df[col].value_counts())
#    print("\n")

# save data_dict
#for room_id, df in train_dict.items():
#    dfg.save_to_csv(df, "data", f"lecture_train_dict")

#for room_id, df in val_dict.items():
#    dfg.save_to_csv(df, "data", f"lecture_val_dict")

#for room_id, df in test_dict.items():
#    dfg.save_to_csv(df, "data", f"lecture_test_dict")
