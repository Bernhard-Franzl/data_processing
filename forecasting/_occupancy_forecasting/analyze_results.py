import os
import ast
import json
import numpy as np
import pandas as pd
import itertools

import plotly.express as px
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from _occupancy_forecasting.data import load_data
import plotly.io as pio

class StatsLogger():
    
    def __init__(self):
        self.comb_lists = []
        self.model_losses = []
        self.zero_baselines = []
        self.naive_baselines = []
        self.avg_baselines = []
        self.dataset_types = []
        self.loss_types = []
     
    def reset(self):
        self.comb_lists = []
        self.model_losses = []
        self.zero_baselines = []
        self.naive_baselines = []
        self.avg_baselines = []
        self.dataset_types = []
        self.loss_types = []
        
    def handle_array_types(self, array_type, array, dataset, loss_type):
        
        if array_type == "Combinations":
                        
            self.dataset_types.extend(np.repeat(dataset, len(array)))
            self.loss_types.extend(np.repeat(loss_type, len(array)))
            
            self.comb_lists.extend(array)
            
        elif array_type == "Model Losses":
            self.model_losses.extend(array)
            
        elif array_type == "BL zero Losses":
            self.zero_baselines.extend(array)
            
        elif array_type == "BL naive Losses":
            self.naive_baselines.extend(array)

        elif array_type == "BL avg Losses":
            self.avg_baselines.extend(array)

        else:
            print(array_type, array)
            raise ValueError("array_type not recognized")
            
    def return_dataframe(self, run_id):
        dataframe_dict = {
            "run_id": run_id,
            "dataset": self.dataset_types,
            "loss_type": self.loss_types,
            "combinations": self.comb_lists,
            "model_losses": self.model_losses,
        }
        if len(self.zero_baselines) > 0:
            dataframe_dict["zero_baselines"] = self.zero_baselines
        
        if len(self.naive_baselines) > 0:
            dataframe_dict["naive_baselines"] = self.naive_baselines
            
        if len(self.avg_baselines) > 0:
            dataframe_dict["avg_baselines"] = self.avg_baselines
        
        return pd.DataFrame(dataframe_dict)
    
class ResultsAnalyis:
    
    def __init__(self, path_to_data, dfguru):
        
        self.path_to_data = path_to_data
        self.helper_path = os.path.join(path_to_data, "helpers")
        
        self.logger = StatsLogger()
        self.dfg = dfguru
        
        self.fundamental_features = ["occrate", "avgocc", "coursenumber", 'lecture', "exam", "tutorium_test_cancelled", "registered", "type", "studyarea", "dow", "hod", "weather"]
        
        self.fundamental_feature_combinations = self.generate_feature_combinations(self.fundamental_features)
        

    ####### Loading results and parsing them #######
    def load_results(self, path_to_resultsfile):
        
        with open(path_to_resultsfile, "r") as f:
            lines = f.readlines()
            line_str = "".join(lines)
            
        list_of_runs = line_str.split("\n\n\n")
        list_of_runs = [run for run in list_of_runs if run != ""][:-1]
        
        return list_of_runs
    
    def parse_run(self, run_corrected):
        
        for elem in run_corrected:
            
            by_bar = elem.split("|")

            if len(by_bar) == 2:
                dataset = by_bar[0].split(":")[1].strip()
                loss_type = by_bar[1].split(":")[1].strip()

            elif len(by_bar) == 1:
                
                if by_bar[0] == "":
                    continue
                
                array_type, array = by_bar[0].split(":")
                array_type = array_type.strip()
                
                if array_type == "Hyperparameters":
                    continue
                
                array = array.strip()
                array = np.array(ast.literal_eval(array))
                self.logger.handle_array_types(array_type, array, dataset, loss_type)
                    
    def parse_list_of_runs(self, list_of_runs):
        
        list_of_dfs = []
        for run in list_of_runs:
            
            splitted_run = run.split("\n")
            # filter out empty strings
            splitted_run = [elem for elem in splitted_run if elem != ""]

            # correct split such that \n does not happen inside and elements
            splitted_run_corrected = []
            for elem in splitted_run[2:]:
                
                by_bar = elem.split("|")
                if len(by_bar) == 2:
                    splitted_run_corrected.append(elem)
                    
                elif len(by_bar) == 1:
                    if by_bar[0] == "":
                        continue

                    if len(by_bar[0].split(":")) == 2:
                        splitted_run_corrected.append(elem)
                    else:
                        splitted_run_corrected[-1] = splitted_run_corrected[-1] + elem
                        
                else:
                    raise
            
            run_id = int(splitted_run[0].split(" ")[2])
            #if (run_id < 13) or (run_id > 16):
            #    continue
            self.logger.reset()
            
            self.parse_run(splitted_run_corrected)
            
            run_df = self.logger.return_dataframe(run_id)
            
            list_of_dfs.append(run_df)
            
        df_results = pd.concat(list_of_dfs).reset_index(drop=True)
        
        return df_results
    
    def add_hyperparameters(self, parsed_results, path_to_checkpoints):
        
        dataframe = parsed_results.copy(deep=True)
        for idx, row in dataframe.iterrows():
            
            comb = row["combinations"]
    
            comb_path = os.path.join(path_to_checkpoints, f"run_{comb[0]}/comb_{comb[1]}")
            hyperparameters_path = os.path.join(comb_path, "hyperparameters.json")

            hyperparameters = json.load(open(hyperparameters_path, "r"))
            
            # overwrite combinations with tuple of run_id and comb_id
            dataframe.at[idx, "combinations"] = (row["run_id"], comb[1])
            
            # add all hyperparameters to the dataframe
            for key, value in hyperparameters.items():
                dataframe.at[idx, key] = str(value)
        
        return dataframe
    
    def data_preparation(self, paths_to_results, paths_to_checkpoints):
        
        list_of_parsed_results = []
        for path_to_resultsfile, path_to_checkpoints in zip(paths_to_results, paths_to_checkpoints):
            list_of_runs = self.load_results(path_to_resultsfile)
            parsed_results = self.parse_list_of_runs(list_of_runs)
            parsed_results = self.add_hyperparameters(parsed_results, path_to_checkpoints)
            list_of_parsed_results.append(parsed_results)
            
        return pd.concat(list_of_parsed_results).reset_index(drop=True)
        #return parsed_results
    
    def filter_clean_pivot(self, dataframe, filter_dict):
        
        filtered_dataframe = self.filter_dataframe_by_dict(dataframe, filter_dict)
        
        list_of_features = list(filter_dict.keys()) + ["features", "run_id"]
        print(list_of_features)
         
        cleaned_dataframe = filtered_dataframe[list_of_features + ["model_losses"]]       
        cleaned_dataframe = cleaned_dataframe.drop_duplicates(subset=list_of_features)
        
        print(cleaned_dataframe.shape)
        
        pivot_dataframe = cleaned_dataframe.pivot(index=['features', 'run_id'], columns="loss_type", values='model_losses')
        #pivot_dataframe.columns = ['features', 'run_id', 'loss_type', 'test_loss', 'val_loss']
        
        return pivot_dataframe
    
    ###### Save & Load Dataframes ######
    def save_dataframe(self, dataframe, path_to_repo, filename):
        return self.dfg.save_to_csv(dataframe, path_to_repo, filename)
    
    def load_dataframe(self, path_to_repo, filename):
        return self.dfg.load_dataframe(path_to_repo, filename)
    
    ###### Filter functions ######
    def filter_dataframe_by_column_value(self, dataframe, column, value):
        # also accept list as values:
        # check if value is type list
        if isinstance(value, list):
            return dataframe[dataframe[column].isin(value)].reset_index(drop=True)
        else:
            return dataframe[dataframe[column] == value].reset_index(drop=True)
    
    def filter_dataframe_by_dict(self, dataframe, filter_dict):
        for key, value in filter_dict.items():
            dataframe = self.filter_dataframe_by_column_value(dataframe, key, value)
        return dataframe
    
    ##### Helper functions ######
    def split_to_fundamental_features(self, feature):
        feature_set = set()
        for fundamental_feature in self.fundamental_features:
            if fundamental_feature in feature:
                feature_set.add(fundamental_feature)
        return feature_set
    
    def generate_feature_combinations(self, feature_set):
        # delete occrate from feature set
        feature_set_copy = feature_set.copy()
        feature_set_copy.remove("occrate")
        
        all_combs = []
        for i in range(0, len(feature_set_copy)+1):
            combs = list(itertools.combinations(feature_set_copy, i))
            all_combs.extend(combs)
            
        all_combs = [["occrate"]+list(comb) for comb in all_combs]
        all_combs = [set(comb) for comb in all_combs]
        
        return all_combs
        
    def top_k_rows(self, dataframe, target_column, k, ascending):
        return dataframe.sort_values(target_column, ascending=ascending).head(k)

    ##### Grouping functions ######
    def group_by_features(self, dataframe, target_column):
        
        return dataframe.groupby('features').agg(
            #mean_validation_mae=('val_loss', 'mean'),
            #std_validation_mae=('val_loss', 'std'),
            mean_loss=(target_column, 'mean'),
            std_loss=(target_column, 'std')
        ).reset_index().sort_values("mean_loss").reset_index(drop=True)
        
    def group_by_subfeatures(self, dataframe, target_column):
        
        df = dataframe.copy(deep=True)
        
        all_features = set(list(df['features']))
        for feature in all_features:
            df[feature] = df['features'].apply(lambda x: feature in x)

        # Group by individual features and calculate mean/std
        feature_analysis = {}
        for feature in all_features:
            df_loc = df.loc[df[feature], target_column]
            feature_analysis[feature] = {
                #'mean_validation_mae': dataframe.loc[dataframe[feature], 'val_loss'].mean(),
                #'std_validation_mae': dataframe.loc[dataframe[feature], 'val_loss'].std(),
                'mean_loss': df_loc.mean(),
                'std_loss': df_loc.std(),
                'count': len(df_loc)
            }

        return_df = pd.DataFrame.from_dict(feature_analysis, orient='index').sort_index().reset_index().rename(columns={"index": "features"})
        return return_df.sort_values("mean_loss").reset_index(drop=True)
    
    def group_by_fundamental_features(self, dataframe, target_column):
        
        df = dataframe.copy(deep=True)
        
        all_features = self.fundamental_features
        for feature in all_features:
            df[feature] = df['features'].apply(lambda x: feature in x)

        # Group by individual features and calculate mean/std
        feature_analysis = {}
        for feature in all_features:
            df_loc = df.loc[df[feature], target_column]
            feature_analysis[feature] = {
                #'mean_validation_mae': dataframe.loc[dataframe[feature], 'val_loss'].mean(),
                #'std_validation_mae': dataframe.loc[dataframe[feature], 'val_loss'].std(),
                'mean_loss': df_loc.mean(),
                'std_loss': df_loc.std(),
                'count': len(df_loc)
            }

        return_df = pd.DataFrame.from_dict(feature_analysis, orient='index').sort_index().reset_index().rename(columns={"index": "features"})
        return return_df.sort_values("mean_loss").reset_index(drop=True)
    
    def group_by_component_features(self, dataframe, target_column):
        
        df = dataframe.copy(deep=True)

        df["fundamental_features"] = df["features"].apply(lambda x: self.split_to_fundamental_features(x))

        # apply function to all rows
        for fundamental_comb in self.fundamental_feature_combinations:
            name = ",".join(sorted(list(fundamental_comb)))
            df[name] = df["fundamental_features"].apply(lambda x: fundamental_comb.issubset(x))

        feature_analysis = {}
        for fundamental_comb in self.fundamental_feature_combinations:
            name = ",".join(sorted(list(fundamental_comb)))
            df_loc = df.loc[df[name], target_column]
            feature_analysis[name] = {
                'mean_loss': df_loc.mean(),
                'std_loss': df_loc.std(),
                'count': len(df_loc)
            }
            
        return_df = pd.DataFrame.from_dict(feature_analysis, orient='index').sort_index().reset_index().rename(columns={"index": "features"})

        return return_df.sort_values("mean_loss").reset_index(drop=True)
        
    ##### Analysis Functions ######
    def get_binary_features(self, dataframe):   

        binary_features = pd.DataFrame()
        for feature in self.fundamental_features:
            binary_features[feature] = dataframe['features'].apply(lambda x: 1 if feature in x else 0)
        
        return binary_features   
    
    def correlate_features_with_loss(self, dataframe, target_column):
        
        binary_features = self.get_binary_features(dataframe)
        target_metrics = dataframe[target_column]
        
        correlations = binary_features.corrwith(target_metrics)
        return correlations
    
    def calc_feature_inclusion(self, dataframe, feature, target):
        # feature inclusion
        dataframe[feature] = dataframe['features'].apply(lambda x: feature in x)
        feature_inclusion = dataframe.groupby(feature)[target].agg(['mean', 'std']).reset_index()
        
        # impact
        with_feature = feature_inclusion[feature_inclusion[feature] == True]
        without_feature = feature_inclusion[feature_inclusion[feature] == False]
        impact = with_feature["mean"].values - without_feature["mean"].values
        
        return feature_inclusion, impact
    
    def calc_benchmark_improvement(self, dataframe, feature, target):
        # the benchmark is the feature: "occrate"
        benchmark = dataframe[dataframe['features'] == "occrate"]
        
        dataframe[feature] = dataframe['features'].apply(lambda x: feature in x)
        feature_inclusion = dataframe.groupby(feature)[target].agg(['mean', 'std']).reset_index()

        feature_inclusion["benchmark"] = benchmark[target].values[0]
        feature_inclusion["improvement"] = feature_inclusion["benchmark"] - feature_inclusion["mean"]

        return feature_inclusion
    
    def calc_linear_regression(self, dataframe, target_column, intercept):
        
        binary_features = self.get_binary_features(dataframe)
        X = binary_features
        y = dataframe[target_column]
        
        if intercept:
            model = LinearRegression().fit(X, y)
            reg_results = pd.Series(np.append(model.coef_, model.intercept_), index=list(X.columns) + ['intercept'])
        
        else:
            model = LinearRegression(fit_intercept=False).fit(X, y)
            reg_results = pd.Series(model.coef_, index=X.columns)
        
        return reg_results, model, model.predict(X)
    
    def prepare_data_decision_tree(self, dataframe, target_column, train_test):
        
        binary_features = self.get_binary_features(dataframe)
        X = binary_features
        y = dataframe[target_column]
        
        if train_test:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        
        else:
            return X, y
    
    def calc_decision_tree(self, X, y, parameters):  
        
        model = DecisionTreeRegressor(random_state=42, **parameters)
        model.fit(X, y)

        # Predict
        y_pred = model.predict(X)
        
        return model, y_pred

    def calc_random_forest(self, dataframe, target_column, parameters=dict()):
      
        binary_features = self.get_binary_features(dataframe)
        X = binary_features
        y = dataframe[target_column]
          
        model = RandomForestRegressor(random_state=42, **parameters)

        model.fit(X, y)
        # Predict
        y_pred = model.predict(X)
        
        return model, y_pred
        
    ##### Plotting Functions ######
    def scatter_plot_feature_group(self, dataframe, x_col, y_col, file_name):
    
        df = dataframe.copy(deep=True)
        
        # Add a column to differentiate the "occrate" feature
        df['highlight'] = df['FeatureSet'].apply(lambda x: 'occrate' if x == ["OccupancyRate"] else 'other')
        
        color_sequence = px.colors.qualitative.Plotly
        # Create the scatter plot
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color='highlight',  # Highlight "occrate" in a different color
            color_discrete_map={'occrate': color_sequence[1], 'other': color_sequence[0]},
            hover_data=['FeatureSet'],  # Show feature names on hover
            #title='Avg. Denormalized MAE VS Avg. Denormalized MSE',
        )

        fig.update_traces(
            hovertemplate=(
            "<b>%{customdata}</b><br>"
            "<b>X:</b> %{x:.6f}<br>"
            "<b>Y:</b> %{y:.6f}<extra></extra>"
            )
        )
        
        # Customize the layout
        fig.update_layout(
            title_font_size=25,
            title_x=0.5,
            template='plotly_white',
            width=1000,
            height=1000,
            margin=dict(l=100, r=50, t=50, b=100),
            xaxis_title="Avg. Denormalized MAE",
            yaxis_title="Avg. Denormalized MSE",
            font=dict(
                size=18,
                color="Black"
            ),
            showlegend=False
        )
        # save figure with high resolution
        #save a figure of 300dpi, with 1.5 inches, and  height 0.75inches
        scale = 4
        dpi = 1000
        pio.write_image(fig, file_name, width=1*dpi, height=1*dpi, scale=scale)

        # Show the plot
        fig.show()
        
    def scatter_plot_target_prediction(self, model_name, y_true, y_pred):
    
        plt.figure(figsize=(10, 6))

        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)  # Perfect prediction line

        plt.xlabel("Actual Values (y_true)")
        plt.ylabel("Predicted Values (y_pred)")

        mse = mean_squared_error(y_true, y_pred)
        plt.title(f"y_true vs y_pred ({model_name}) - MSE: {mse}")

        plt.grid(True)

        plt.show()