import numpy as np
import random
import torch

from _occupancy_forecasting import prepare_data
from _dfguru import DataFrameGuru as DFG
from _occupancy_forecasting import OccFeatureEngineer
dfg = DFG()


for split_by in ["time"]:
    for frequency in ["1h", "1min", "5min", "15min", "30min"]:
        for with_examweek in [True, False]:
        
            #### Control Randomness ####
            np_rng = np.random.default_rng(seed=42)
            torch_rng = torch.Generator()
            torch_rng.manual_seed(42)
            random.seed(42)

            train_dict, val_dict, test_dict = prepare_data(
                path_to_data_dir="data/occupancy_forecasting",
                frequency=frequency,
                feature_list=sorted(list(OccFeatureEngineer.permissible_features)),
                dfguru=dfg,
                rng=np_rng,
                split_by=split_by,
                helpers_path="data/occupancy_forecasting",
                with_examweek=with_examweek,
            )

            if with_examweek:
                add_to_string = "_with-examweek"
            else:
                add_to_string = "_without-examweek"
                
            # save data_dict
            for room_id, df in train_dict.items():
                dfg.save_to_csv(
                    df, 
                    f"data/occupancy_forecasting/freq_{frequency}", 
                    f"room-{room_id}_{split_by}_train_dict" + add_to_string)
            
            for room_id, df in val_dict.items():
                dfg.save_to_csv(
                    df, 
                    f"data/occupancy_forecasting/freq_{frequency}", 
                    f"room-{room_id}_{split_by}_val_dict" + add_to_string)
            
            for room_id, df in test_dict.items():
                dfg.save_to_csv(
                    df, 
                    f"data/occupancy_forecasting/freq_{frequency}", 
                    f"room-{room_id}_{split_by}_test_dict" + add_to_string)
