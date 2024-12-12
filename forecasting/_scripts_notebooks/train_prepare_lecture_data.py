import numpy as np
import random
import torch

from _lecture_forecasting import prepare_data, load_data
from _dfguru import DataFrameGuru as DFG
from _lecture_forecasting import LectureFeatureEngineer

dfg = DFG()

################# Lecture Data ##################
np_rng = np.random.default_rng(seed=42)
torch_rng = torch.Generator()
torch_rng.manual_seed(42)
random.seed(42)

test = True
for n_weeks in [9]:
    
    # for deploying
    #split_by = f"random_{n_weeks}"
    # for training
    split_by = f"time_{n_weeks}"
    
    datasets, indices = prepare_data(
        path_to_data_dir="data/lecture_forecasting",
        feature_list=sorted(list(LectureFeatureEngineer.permissible_features)),
        dfguru = dfg,
        rng = np_rng,
        split_by=split_by,
        test = test,
    )

    if "random" in split_by:
        
        print(f"Saving data for {split_by}")
        dfg.save_to_csv(datasets[0], "data/lecture_forecasting", f"lecture_train_{split_by}")
        dfg.save_to_csv(datasets[1], "data/lecture_forecasting", f"lecture_test_{split_by}")
        np.save(f"data/lecture_forecasting/lecture_train_idx_{split_by}.npy", indices[0])
        np.save(f"data/lecture_forecasting/lecture_val_idx_{split_by}.npy", indices[1])
        
        
    else:
        dfg.save_to_csv(datasets[0], "data/lecture_forecasting", f"lecture_train_{split_by}")
        dfg.save_to_csv(datasets[1], "data/lecture_forecasting", f"lecture_val_{split_by}")
        if test:
            dfg.save_to_csv(datasets[2], "data/lecture_forecasting", f"lecture_test_{split_by}")

#train_df, val_df, test_df = load_data("data/lecture_forecasting", dfguru=dfg)

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