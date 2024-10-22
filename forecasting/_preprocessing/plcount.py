import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import time
from pandas.tseries.offsets import DateOffset

# class PLCount that implements the algorithm explained in the paper
# make sure to always use the vectorized version of the algorithm
class PLCount():
    
    def __init_(self):
        pass
        
    def initialize_algorithm(self, n, m):
        M = np.zeros((n, m+1))
        N = np.zeros((n, m+1))
        M[0,0] = 1
        return M, N
         
    def calc_delta(self, dataframe, column):
        return dataframe[column].diff().fillna(0)
    
    def calc_sigma(self, dataframe, column, multiplier):
        sigma = dataframe[column].apply(lambda x : np.sqrt(np.abs(x))*multiplier)
        #sigma = dataframe[column].apply(lambda x : np.abs(x))
        try:
            sigma = sigma.replace(0, min(sigma[sigma > 0]))
        except:
            sigma = sigma.replace(0, 1)
            
        return sigma
    
    def probability_function(self, c_j, delta_c, sigma):
        exponent = -(c_j - delta_c)**2 / (2 * sigma**2)
        normalizer = 1 / (sigma * np.sqrt(2 * np.pi))
        return normalizer * np.exp(exponent)
    
    def calculate_probability_matrix(self, M, N, delta_array, sigma_array):
        
        for i in range(1, M.shape[0]): # time
            delta_c_i = delta_array[i]
            sigma_i = sigma_array[i]
                
            for j in range(0, M.shape[1]): # count
                
                listy = [self.probability_function(j-k, delta_c_i, sigma_i) * M[i-1, k] for k in range(0, M.shape[1])]
                
                k_max = np.argmax(listy)
                
                M[i, j] = listy[k_max]
                N[i, j] = k_max

            print(N[i])
            # normalize row  
            M[i] = M[i] / sum(M[i])
            
        return M, N
    
    def backtracking_zero_init(self, M, N):
        CC_t_n = 0
        occupancy_estimates = np.zeros(M.shape[0])

        for i in range(M.shape[0]-1, 0, -1):
            
            CC_t_n1 = N[i, int(CC_t_n)]
            occupancy_estimates[i-1] = CC_t_n1
            
            CC_t_n = CC_t_n1
        
        return occupancy_estimates
    
    def run_algorithm(self, n, m, delta_array, sigma_array):
        
        M, N = self.initialize_algorithm(n, m)
        
        M, N = self.calculate_probability_matrix(M, N, delta_array, sigma_array)
        
        occupancy_estimates = self.backtracking_zero_init(M, N)
        
        return occupancy_estimates
    
    ##############################
    def probability_function(self, c_j, delta_c, sigma):
        exponent = -(c_j - delta_c)**2 / (2 * sigma**2)
        normalizer = 1 / (sigma * np.sqrt(2 * np.pi))
        return normalizer * np.exp(exponent)
    
    def calculate_probability_matrix_vectorized(self, M, N, delta_array, sigma_array):
        
        M_shape = M.shape
        
        for i in range(1, M_shape[0]): # time
            delta_c_i = delta_array[i]
            sigma_i = sigma_array[i]

            j_array = np.arange(0, M_shape[1]).reshape(-1,1)
            k_array = np.arange(0, M_shape[1]).reshape(1,-1)
            
            j_k_matrix = j_array - k_array
            exponent = -(j_k_matrix - delta_c_i)**2 / (2 * sigma_i**2)
            normalizer = 1 / (sigma_i * np.sqrt(2 * np.pi))
            M_i = normalizer * np.exp(exponent) * M[i-1].reshape(1,-1)
            
            k_max = np.argmax(M_i, axis=1)
            
            # sample from M_i instead of using argmax
            print((k_max-np.array([np.argmax(M_i[i]) for i in range(len(M_i))])).sum())
 
            #print((k_max-np.array([np.random.choice(np.arange(M_i.shape[1]), p=M_i[i]) for i in range(len(M_i))])).sum())
            #k_max = np.random.choice(np.arange(M_shape[1]), p=M_i)
            
            N[i] = k_max
            M[i] = M_i[np.arange(M_shape[1]), k_max]
                        
            # normalize row  
            M[i] = M[i] / sum(M[i])
        raise
        return M, N
    
    def run_algorithm_vectorized(self, n, m, delta_array, sigma_array):
        
        M, N = self.initialize_algorithm(n, m)

        M, N = self.calculate_probability_matrix_vectorized(M, N, delta_array, sigma_array)
        
        occupancy_estimates = self.backtracking_zero_init(M, N)
        
        return occupancy_estimates
    
    def run_on_whole_dataset(self, dataframe, data_handler, frequency, params):
        
        occupancy_count_list = []
        day_list = list(pd.Series(1, dataframe['datetime']).resample("D").sum().index)
        
        for timestamp in tqdm(day_list):
            
            df_filtered = data_handler.filter_by_timestamp(dataframe, "datetime",
                                                  timestamp, timestamp + DateOffset(days=1))
    
            if df_filtered.empty:  
                idx = pd.date_range(start=timestamp, 
                                    end=timestamp + DateOffset(days=1), 
                                    freq=frequency, inclusive="both")
                occupancy_counts = pd.DataFrame(data=0, 
                                                index=idx, 
                                                columns=["CC", "CC_estimates"]).reset_index().rename(columns={"index": "datetime"})
                occupancy_count_list.append(occupancy_counts)    
                
            else:
                occ_counts_raw = data_handler.calc_occupancy_count(df_filtered, "datetime", frequency)
                
                occ_res = occ_counts_raw.copy()
                occ_res["CC_estimates"] = 0
                
                if params["filtering_params"]["discard_samples"]:
                    
                    lb = params["filtering_params"]["discard_times"][0].split(":")
                    ub = params["filtering_params"]["discard_times"][1].split(":")
                    lb = time(hour=int(lb[0]), minute=int(lb[1]), second=int(lb[2]))
                    ub = time(hour=int(ub[0]), minute=int(ub[1]), second=int(ub[2]))
                    # filter lb and ub
                    filter_mask = (occ_counts_raw["datetime"].dt.time >= lb) & (occ_counts_raw["datetime"].dt.time <= ub)
                
                else:        
                    filter_mask = (occ_counts_raw["datetime"].dt.time >= time(0,0,0))        
                
                occ_counts_pl = occ_counts_raw[filter_mask].reset_index(drop=True)

                occ_counts_pl["delta_CC"] = self.calc_delta(occ_counts_pl, "CC")
                occ_counts_pl["sigma"] = self.calc_sigma(occ_counts_pl, "delta_CC", params["plcount_params"]["sigma_factor"])
            
                cc_max = occ_counts_pl.CC.max()
                m = int(cc_max + (cc_max*params["plcount_params"]["cc_max_factor"]))
                n = len(occ_counts_pl.datetime)
                
                estimates = self.run_algorithm_vectorized(n, m, occ_counts_pl["delta_CC"], occ_counts_pl["sigma"])
                occ_res.loc[filter_mask, "CC_estimates"] = estimates       

                occupancy_count_list.append(occ_res)
    
        return occupancy_count_list
