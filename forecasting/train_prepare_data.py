import numpy as np
import random
import torch

from _forecasting import prepare_data, load_data_dicts, prepare_data_lecture, load_data_lecture
from _dfguru import DataFrameGuru as DFG
from _forecasting import OccFeatureEngineer, LectureFeatureEngineer
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




################## Lecture Data ##################
#np_rng = np.random.default_rng(seed=42)
#torch_rng = torch.Generator()
#torch_rng.manual_seed(42)
#random.seed(42)

#test = False
#for n_weeks in range(2, 13):
    
#    split_by = f"random_{n_weeks}"
#    datasets, indices = prepare_data_lecture(
#        path_to_data_dir="data",
#        feature_list=sorted(list(LectureFeatureEngineer.permissible_features)),
#        dfguru = dfg,
#        rng = np_rng,
#        split_by=split_by,
#        test = test,
#    )

#    if "random" in split_by:
#        print(f"Saving data for {split_by}")
#        dfg.save_to_csv(datasets, "data", f"lecture_data_{split_by}")
#        np.save(f"data/lecture_train_idx_{split_by}.npy", indices[0])
#        np.save(f"data/lecture_val_idx_{split_by}.npy", indices[1])
#        np.save(f"data/lecture_test_idx_{split_by}.npy", indices[2])
        
#    else:
#        dfg.save_to_csv(datasets[0], "data", f"lecture_train_{split_by}")
#        dfg.save_to_csv(datasets[1], "data", f"lecture_val_{split_by}")
#        if test:
#            dfg.save_to_csv(datasets[2], "data", f"lecture_test_{split_by}")

#train_df, val_df, test_df = load_data_lecture("data", dfguru=dfg)

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
