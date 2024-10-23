import torch
import torch.nn as nn
import numpy as np
import json

class SimpleOccDenseNet(nn.Module):
    
    def __init__(self, hyperparameters):
        super().__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.hyperparameters = hyperparameters

        self.x_horizon = hyperparameters["x_horizon"]
        self.x_size = hyperparameters["x_size"] * self.x_horizon
        self.y_horizon = hyperparameters["y_horizon"]
        self.y_features_size = hyperparameters["y_features_size"] * self.y_horizon 
    
        self.output_size = hyperparameters["y_size"] * hyperparameters["y_horizon"]
    
        self.hidden_size = hyperparameters["hidden_size"]
        
        self.occcount = hyperparameters["occcount"]
        self.batch_size = hyperparameters["batch_size"]
        
        #self.room_encoding = torch.eye(2).to(self.device)
        #self.room_enc_size = self.room_encoding.shape[1]
        self.enc_size = 0
        
        if "coursenumber" in hyperparameters["features"]:
            
            with open("data/helpers_occpred.json", "r") as f:
                self.helper = json.load(f)       
        
            self.enc_dim= 5
            course_numbers = self.helper["course_numbers"]
            weights = torch.zeros(len(course_numbers), self.enc_dim)
            self.course_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
            
            self.enc_size = self.enc_dim*self.x_horizon + self.enc_dim*self.y_horizon
          
        self.input_size = self.x_size + self.y_features_size + self.enc_size
        
        self.model = nn.Sequential()
        if len(self.hidden_size) > 1:
            self.model.add_module("input_layer", nn.Linear(self.input_size, self.hidden_size[0]))
            self.model.add_module("relu_0", nn.ReLU())
            
            for i in range(0, len(self.hidden_size)-1):
                self.model.add_module(f"hidden_layer_{i}", nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))
                self.model.add_module(f"relu_{i+1}", nn.ReLU())
            
            self.model.add_module("output_layer", nn.Linear(self.hidden_size[-1], self.output_size))  
        
        else:
            if len(self.hidden_size) == 0:
                self.model.add_module("input_layer", nn.Linear(self.input_size, self.output_size))
            else:
                self.model.add_module("input_layer", nn.Linear(self.input_size, self.hidden_size[0]))
                self.model.add_module("relu_0", nn.ReLU())
                self.model.add_module("output_layer", nn.Linear(self.hidden_size[0], self.output_size))
            
        if self.occcount:
            if self.hyperparameters["differencing"] in ["sample", "whole"]:
                pass
            else:
                self.model.add_module("acti_out", nn.ReLU())

        else:
            if self.hyperparameters["differencing"] in ["sample", "whole"]:
                self.model.add_module("acti_out", nn.Tanh())
                #print("Tanh added")
            else:
                if self.hyperparameters["criterion"] == "CE":
                    print("nothing added")
                else:   
                    self.model.add_module("acti_out", nn.Identity())
   
    def forward(self, x, y_features, room_id=None):
        
        if "coursenumber" in self.hyperparameters["features"]:
            
            X_course_emb = self.course_embedding(room_id[:, :self.x_horizon])
            y_course_emb = self.course_embedding(room_id[:, self.x_horizon:])
            
            course_emb = torch.cat((X_course_emb, y_course_emb), 1)

            course_emb = course_emb.view(self.batch_size, -1)
            x = x.view(self.batch_size, -1)
            y_features = y_features.view(self.batch_size, -1)
            
            input = torch.cat((x, y_features, course_emb), 1)
        
        else:
            x = x.view(self.batch_size, -1)
            y_features = y_features.view(self.batch_size, -1)
            input = torch.cat((x, y_features), 1)

        out = self.model(input)

        return out
    
    def forecast_iter(self, x, y_features, len_y, room_id=None):
        
        
        if "coursenumber" in self.hyperparameters["features"]:

            X_course_emb = self.course_embedding(room_id[:self.x_horizon])
            y_course_emb = self.course_embedding(room_id[self.x_horizon:])
            

        predicitons = []
        
        for i in range(0, len_y, self.y_horizon):
            
            x_flattened  = x.view(-1)
            
            y_feat_i = y_features[i:i+self.y_horizon]
            
            if y_feat_i.shape[0] < self.y_horizon:
                break
            
            y_feat_i_flattend = y_feat_i.reshape(-1)
            
            if "coursenumber" in self.hyperparameters["features"]:
                
                y_emb_i = y_course_emb[i:i+self.y_horizon]
                x_emb_i = X_course_emb
                
                course_emb = torch.cat((x_emb_i, y_emb_i))
                course_emb = course_emb.view( -1)
                
                input = torch.cat((x_flattened, y_feat_i_flattend, course_emb))
            
            else:

                input = torch.cat((x_flattened, y_feat_i_flattend))
            
            out = self.model(input)

            predicitons.append(out[:, None])

            if self.hyperparameters["include_x_features"]:
                x_new = torch.cat((out[:, None], y_feat_i), dim=1)
            else:
                x_new = out[:, None]
                
            x = torch.cat((x, x_new), dim=0)[self.y_horizon:]
            
            if "coursenumber" in self.hyperparameters["features"]:
                X_course_emb = torch.cat((X_course_emb, y_emb_i))[self.y_horizon:]
        
        if len(predicitons) == 0:
            return torch.tensor([])
        else:
            return torch.cat(predicitons)
    
class OccDenseNet(nn.Module):
    
    def __init__(self, hyperparameters):
        super().__init__()
        
        self.hyperparameters = hyperparameters

        self.x_size = hyperparameters["x_size"] * hyperparameters["x_horizon"]
        self.y_features_size = hyperparameters["y_features_size"] * hyperparameters["y_horizon"]
        self.output_size = hyperparameters["y_size"] * hyperparameters["y_horizon"]
        
        
        self.hidden_size = hyperparameters["hidden_size"]
        
        
        self.occcount = hyperparameters["occcount"]
        self.batch_size = hyperparameters["batch_size"]
            
            
        self.model = nn.Sequential()
        if len(self.hidden_size) > 1:
            self.model.add_module("input_layer", nn.Linear(self.x_size, self.hidden_size[0]))
            self.model.add_module("relu_0", nn.ReLU())
            
            for i in range(0, len(self.hidden_size)-1):
                self.model.add_module(f"hidden_layer_{i}", nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))
                self.model.add_module(f"relu_{i+1}", nn.ReLU())
            
            self.model.add_module("output_layer", nn.Linear(self.hidden_size[-1], 15))  
        else:
            if len(self.hidden_size) == 0:
                self.model.add_module("input_layer", nn.Linear(self.x_size, self.output_size))
            else:
                self.model.add_module("input_layer", nn.Linear(self.x_size, self.hidden_size[0]))
                self.model.add_module("relu", nn.ReLU())
                self.model.add_module("output_layer", nn.Linear(self.hidden_size[0], self.output_size))
            
        if self.occcount:
            self.model.add_module("", nn.ReLU())
        else:
            self.model.add_module("sigmoid", nn.Sigmoid())
            
    def forward(self, x, y_features):
            
            x = x.view(self.batch_size, -1)
            y_features = y_features.view(self.batch_size, -1)
            input = torch.cat((x, y_features), 1)
    
            x = self.model(input)
            
            return x  
     
     
        
class SimpleOccLSTM(torch.nn.Module):
    
    def __init__(self, hyperparameters, **kwargs):
        
        super().__init__()
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.hyperparameters = hyperparameters

        self.x_size = hyperparameters["x_size"]
        self.x_horizon = hyperparameters["x_horizon"]
        
        self.y_horizon = hyperparameters["y_horizon"]
        self.y_features_size = hyperparameters["y_features_size"]
        self.y_size = hyperparameters["y_size"]
        
        self.output_size = hyperparameters["y_size"] * hyperparameters["y_horizon"]
    
        self.hidden_size = hyperparameters["hidden_size"]
        
        self.batch_size = hyperparameters["batch_size"]
        ############ Model Definition ############
        
        self.enc_size = 0
        self.bdir_factor = 2 if hyperparameters["bidirectional"] else 1
        
        if "coursenumber" in hyperparameters["features"]:
            
            with open("data/helpers_occpred.json", "r") as f:
                self.helper = json.load(f)       
        
            self.enc_dim = 5
            course_numbers = self.helper["course_numbers"]
            weights = torch.zeros(len(course_numbers), self.enc_dim)
            self.course_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
            
            self.enc_size = self.enc_dim
          
        self.input_size = self.x_size + self.enc_size
        
        # lstm layer
        self.lstm = torch.nn.LSTM(self.input_size,  self.hidden_size[0], 
                                  batch_first=True, bidirectional=hyperparameters["bidirectional"],
                                  num_layers=hyperparameters["num_layers"])
          
        # fc at end
        self.fc_end = nn.Sequential()
        if len(self.hidden_size) == 2:
            raise
            self.fc_end.add_module("input_layer", nn.Linear(self.hidden_size[0], self.hidden_size[1]))
            
            if hyperparameters["layer_norm"]:
                self.fc_end.add_module("layer_norm", nn.LayerNorm(self.hidden_size[1]))
                
            self.fc_end.add_module("relu_0", nn.ReLU())
            self.fc_end.add_module("output_layer", nn.Linear(self.hidden_size[1], self.y_size))
        
        else:
            self.fc_end.add_module("input_layer", nn.Linear(self.hidden_size[0]*self.bdir_factor, self.y_size))
            
        if "forget_gate" in hyperparameters:
            if not hyperparameters["forget_gate"]:
                for name, param in self.lstm.named_parameters():
                    if 'bias_ih' in name or 'bias_hh' in name:
                        bias_size = param.size(0) // 4 
                        param.data[bias_size:2*bias_size].fill_(25)
                    
        # last activation function
        if hyperparameters["occcount"]:
            self.last_activation = nn.ReLU()    
        else:
            self.last_activation = nn.Identity()
            
        ## freeze h_0 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def forward(self, x, y_features, room_id=None):
        
        if "coursenumber" in self.hyperparameters["features"]:
            
            X_course_emb = self.course_embedding(room_id[:, :self.x_horizon])
            y_course_emb = self.course_embedding(room_id[:, self.x_horizon:])
            
            x = torch.cat((x, X_course_emb), -1)
            y_features = torch.cat((y_features, y_course_emb), -1)

        out, (h_n, c_n) = self.lstm(x)       
        
        y_t = self.last_activation(self.fc_end(out[:, -1, :]))  
        y_t = y_t[:, None, :]
        
        pred_list = []
        for i in range(0, self.hyperparameters["y_horizon"]):
            
            y_feat_t = y_features[:, i, :]
            y_in = torch.cat((y_t, y_feat_t[:, None, :]), 2)
            
            h_t1, (h_n, c_n) = self.lstm(y_in, (h_n, c_n))
            y_t = self.last_activation(self.fc_end(h_t1))

            pred_list.append(y_t)
        
        pred_list = torch.cat(pred_list, dim=1).squeeze(-1)
        
        return pred_list
    
    def forecast_iter(self, x, y_features, len_y, room_id):
        
        x = x[None, :]
        y_features = y_features[None, :]
        
        if "coursenumber" in self.hyperparameters["features"]:
            
            X_course_emb = self.course_embedding(room_id[:self.x_horizon])[None, :]
            y_course_emb = self.course_embedding(room_id[self.x_horizon:])[None, :]
            
            x = torch.cat((x, X_course_emb), -1)
            y_features = torch.cat((y_features, y_course_emb), -1)


        out, (h_n, c_n) = self.lstm(x)       
        y_t = self.last_activation(self.fc_end(out[:, -1, :]))[:, None, :]
        
        
        pred_list = []
        for i in range(0, len_y):
            
            y_feat_i = y_features[:, i, :][None, :]
            y_t_in = torch.cat((y_t, y_feat_i), 2)
            
            out, (h_n, c_n) = self.lstm(y_t_in, (h_n, c_n))

            y_t = self.last_activation(self.fc_end(out))

            pred_list.append(y_t.squeeze(-1))
        
        return torch.cat(pred_list)

class EncDecOccLSTM(torch.nn.Module):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self, hyperparameters, **kwargs):
        
        super().__init__()
        
        self.hyperparameters = hyperparameters
        
        self.x_horizon = hyperparameters["x_horizon"]
        self.y_horizon = hyperparameters["y_horizon"]
        
        self.output_size = hyperparameters["y_size"] * self.y_horizon

        self.hidden_size = hyperparameters["hidden_size"]
        self.bidir_factor = 2 if hyperparameters["bidirectional"] else 1
        
        encoding_dim = 0
        if "coursenumber" in hyperparameters["features"]:
            
            with open("data/helpers_occpred.json", "r") as f:
                helper = json.load(f)
                course_numbers = helper["course_numbers"]
                del helper  
        
            encoding_dim = hyperparameters["course_encoding_dim"]
            weights = torch.zeros(len(course_numbers), encoding_dim)
            self.course_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        
        # lstm encoder
        enc_input_size = hyperparameters["x_size"] + encoding_dim
        self.encoder_lstm = torch.nn.LSTM(enc_input_size, 
                                          self.hidden_size[0], 
                                          batch_first=True,
                                          bidirectional=hyperparameters["bidirectional"],
                                          num_layers=hyperparameters["num_layers"])
        
        # lstm decoder
        dec_input_size = hyperparameters["y_features_size"] + encoding_dim
        self.decoder_lstm = torch.nn.LSTM(dec_input_size, 
                                          self.hidden_size[0], 
                                          batch_first=True, 
                                          bidirectional=hyperparameters["bidirectional"],
                                          num_layers=hyperparameters["num_layers"])
        
        # predictor
        if len(self.hidden_size) == 2:
            raise
            #self.fc_end = nn.Sequential()
            #self.fc_end.add_module("input_layer", nn.Linear(self.hidden_size[0]*self.bidir_factor, self.hidden_size[1]))
            #self.fc_end.add_module("relu_0", nn.ReLU())
            #self.fc_end.add_module("output_layer", nn.Linear(self.hidden_size[1], self.y_size  )) 
        self.fc_end = nn.Linear(self.hidden_size[0]*self.bidir_factor, hyperparameters["y_size"])

        # additive noise
        self.add_noise = False
        if hyperparameters["additive_noise"] > 0:
            self.additive_noise = torch.distributions.normal.Normal(0, hyperparameters["additive_noise"])
            self.add_noise = True
            
        
    def forward(self, x, y_features, room_id=None):
        
        if self.add_noise:
            # add noise to make it more robust  
            x[:, :, 0] = x[:, :, 0] + self.additive_noise.sample(x[:, :, 0].shape).to(self.device)         
            # remove negative values probably absolute value
            x[:, :, 0] = torch.abs(x[:, :, 0])
        
        if "coursenumber" in self.hyperparameters["features"]:
            
            X_course_emb = self.course_embedding(room_id[:, :self.x_horizon])
            y_course_emb = self.course_embedding(room_id[:, self.x_horizon:])
            
            x = torch.cat((x, X_course_emb), -1)
            y_features = torch.cat((y_features, y_course_emb), -1)
        
        if self.hyperparameters["differencing"] == "whole":
            x[:, :, 0] = x[:, :, 0]*0.5 +  0.5
            raise
        
        _, (_, c_n) = self.encoder_lstm(x)

        h_n = torch.zeros(self.hyperparameters["num_layers"]*self.bidir_factor, x.shape[0],  self.hidden_size[0]).to(self.device)
        out_dec, _ = self.decoder_lstm(y_features, (h_n, c_n))
        
        #pred = [self.fc_end(pred[:, i, :]) for i in range(self.y_horizon)]
        #pred = torch.transpose(torch.stack(pred), 0, 1).squeeze(-1) 
        pred = self.fc_end(out_dec)
        pred = pred.squeeze(-1)
        
        #if self.forward_print_count% 250 == 0:
        #    print(pred.min(), pred.max())
        #self.forward_print_count += 1
        
        return pred  
        
    def forecast_iter(self, x, y_features, len_y, room_id):
        x = x[None, :]
        y_features = y_features[None, :]
        
        if "coursenumber" in self.hyperparameters["features"]:
            X_course_emb = self.course_embedding(room_id[:self.x_horizon])[None, :]
            y_course_emb = self.course_embedding(room_id[self.x_horizon:])[None, :]
    
        predicitons = []
        for i in range(0, len_y, self.y_horizon):
            
            if self.hyperparameters["differencing"] == "whole":
                raise
                x[:, :, 0] = x[:, :, 0]*0.5 +  0.5
                
            y_feat_i = y_features[:, i:i+self.y_horizon]    
            
            if y_feat_i.shape[1] < self.y_horizon:
                break
            
            if "coursenumber" in self.hyperparameters["features"]:
                
                y_emb_i = y_course_emb[:, i:i+self.y_horizon]
                x_emb_i = X_course_emb

                in_1 = torch.cat((x, x_emb_i), -1)
                in_2 = torch.cat((y_feat_i, y_emb_i), -1)
            
            else:
                raise
                in_1 = torch.cat((x, x_emb_i), -1)
                in_2 = torch.cat((y_feat_i, y_emb_i), -1)
             
            _, (_, c_n) = self.encoder_lstm(in_1)

            #h_n = torch.zeros(self.hyperparameters["num_layers"]*self.bidir_factor, x.shape[0],  self.y_size).to(self.device)
            #pred, _ = self.decoder_lstm(in_2, (h_n, c_n))
            #out = pred.squeeze(-1)

            h_n = torch.zeros(self.hyperparameters["num_layers"]*self.bidir_factor, x.shape[0],  self.hidden_size[0]).to(self.device)
            out_dec, _ = self.decoder_lstm(in_2, (h_n, c_n))
            
            #pred = [self.fc_end(out_dec[:, i, :]) for i in range(self.y_horizon)]
            #pred = torch.transpose(torch.stack(pred), 0, 1).squeeze(-1)
            #print(pred.shape)
            
            pred = self.fc_end(out_dec)
            pred = pred.squeeze(-1)

            out = pred
        
            #out_dec, _ = self.decoder_lstm(in_2, (h_n, c_n))
            #pred = [self.last_activation(self.fc_end(out_dec[:, i, :])) for i in range(self.y_horizon)]
            #out = torch.transpose(torch.stack(pred), 0, 1).squeeze(-1) 
            
            predicitons.append(out[:, :, None])

            if self.hyperparameters["include_x_features"]:
                x_new = torch.cat((out[:, :, None], y_feat_i), dim=-1)
            else:
                x_new = out[:, None]
            
            x = torch.cat((x, x_new), dim=1)[:, self.y_horizon:]
              
            if "coursenumber" in self.hyperparameters["features"]:
                X_course_emb = torch.cat((X_course_emb, y_emb_i), dim=1)[:,self.y_horizon:]

        if len(predicitons) == 0:
            return torch.tensor([])
        else:
            return torch.cat(predicitons, dim=1).squeeze(0)
                
    def forecast_iter_old(self, x, y_features, len_y, room_id):
        
        x = x[None, :]
        y_features = y_features[None, :]
        
        if "coursenumber" in self.hyperparameters["features"]:
            
            X_course_emb = self.course_embedding(room_id[:self.x_horizon])[None, :]
            y_course_emb = self.course_embedding(room_id[self.x_horizon:])[None, :]
            
            x = torch.cat((x, X_course_emb), -1)
            y_features = torch.cat((y_features, y_course_emb), -1)
        
        if self.hyperparameters["differencing"] == "whole":
            x[:, :, 0] = x[:, :, 0]*0.5 +  0.5
            
        _, (_, c_n) = self.encoder_lstm(x)

        h_n = torch.zeros(self.hyperparameters["num_layers"]*self.bidir_factor, x.shape[0],  self.hidden_size[0]).to(self.device)
        pred, _ = self.decoder_lstm(y_features, (h_n, c_n))
        
        pred = [self.fc_end(pred[:, i, :]) for i in range(len_y)]
        pred = torch.stack(pred).squeeze(-1) 

        return pred

class EncDecOccLSTM1(torch.nn.Module):
    def __init__(self, hyperparameters, **kwargs):
        
        super().__init__()
        
        self.hyperparameters = hyperparameters
        
        self.x_size = hyperparameters["x_size"]
        self.x_horizon = hyperparameters["x_horizon"]
        
        self.y_features_size = hyperparameters["y_features_size"]
        self.y_horizon = hyperparameters["y_horizon"]
        self.y_size = hyperparameters["y_size"]
    
        self.hidden_size = hyperparameters["hidden_size"]
        
        self.occcount = hyperparameters["occcount"]
        self.batch_size = hyperparameters["batch_size"]
        
        self.enc_size = 0
        self.bidir_factor = 2 if hyperparameters["bidirectional"] else 1
        
        if "coursenumber" in hyperparameters["features"]:
            
            with open("data/helpers_occpred.json", "r") as f:
                self.helper = json.load(f)       
        
            self.enc_dim= 5
            course_numbers = self.helper["course_numbers"]
            weights = torch.zeros(len(course_numbers), self.enc_dim)
            self.course_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
            
            self.enc_size = self.enc_dim
          
          
        self.input_size_1 = self.x_size + self.enc_size
        self.input_size_2 = self.y_features_size + self.enc_size
        self.output_size = self.y_size * self.y_horizon
        
        
        # lstm encoder
        self.encoder_lstm = torch.nn.LSTM(self.input_size_1, 
                                          self.hidden_size[0], 
                                          batch_first=True,
                                          bidirectional=False,
                                          num_layers=hyperparameters["num_layers"])
        
        # lstm decoder
        self.decoder_lstm = torch.nn.LSTM(self.input_size_2, 
                                          self.hidden_size[0]//2, 
                                          batch_first=True, 
                                          bidirectional=hyperparameters["bidirectional"],
                                          num_layers=hyperparameters["num_layers"])
        
        # predictor
        self.pred_lstm = torch.nn.LSTM(self.hidden_size[0]*self.bidir_factor, 
                                        self.hidden_size[0]//4, 
                                        batch_first=True, 
                                        proj_size=1,
                                        num_layers=hyperparameters["num_layers"])
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.forward_print_count = 0

    def forward(self, x, y_features, room_id=None):
    
        if "coursenumber" in self.hyperparameters["features"]:
            
            X_course_emb = self.course_embedding(room_id[:, :self.x_horizon])
            y_course_emb = self.course_embedding(room_id[:, self.x_horizon:])
            
            x = torch.cat((x, X_course_emb), -1)
            y_features = torch.cat((y_features, y_course_emb), -1)
        
        if self.hyperparameters["differencing"] == "whole":
            x[:, :, 0] = x[:, :, 0]*0.5 +  0.5
            
        _, (_, c_n) = self.encoder_lstm(x)

        h_n = torch.zeros(self.hyperparameters["num_layers"]*self.bidir_factor, x.shape[0],  self.hidden_size[0]).to(self.device)
        c_n = c_n.repeat(self.bidir_factor, 1, 1)
        out_dec, _ = self.decoder_lstm(y_features, (h_n, c_n))
        raise
        
        
        pred, _  = self.pred_lstm(out_dec)

        #pred = [self.fc_end(pred[:, i, :]) for i in range(self.y_horizon)]
        #pred = torch.transpose(torch.stack(pred), 0, 1).squeeze(-1) 
        pred = pred.squeeze(-1)
        
        
        #if self.forward_print_count% 250 == 0:
        #    print(pred.min(), pred.max())
        #self.forward_print_count += 1
        
        return pred  
        
    def forecast_iter(self, x, y_features, len_y, room_id):
        x = x[None, :]
        y_features = y_features[None, :]
        
        if "coursenumber" in self.hyperparameters["features"]:
            X_course_emb = self.course_embedding(room_id[:self.x_horizon])[None, :]
            y_course_emb = self.course_embedding(room_id[self.x_horizon:])[None, :]
    
        predicitons = []
        for i in range(0, len_y, self.y_horizon):
            
            if self.hyperparameters["differencing"] == "whole":
                x[:, :, 0] = x[:, :, 0]*0.5 +  0.5
                
            y_feat_i = y_features[:, i:i+self.y_horizon]    
            
            if y_feat_i.shape[1] < self.y_horizon:
                break
            
            if "coursenumber" in self.hyperparameters["features"]:
                
                y_emb_i = y_course_emb[:, i:i+self.y_horizon]
                x_emb_i = X_course_emb

                in_1 = torch.cat((x, x_emb_i), -1)
                in_2 = torch.cat((y_feat_i, y_emb_i), -1)
            
            else:
                raise
                in_1 = torch.cat((x, x_emb_i), -1)
                in_2 = torch.cat((y_feat_i, y_emb_i), -1)
            
            
            _, (_, c_n) = self.encoder_lstm(in_1)

            #h_n = torch.zeros(self.hyperparameters["num_layers"]*self.bidir_factor, x.shape[0],  self.y_size).to(self.device)
            #pred, _ = self.decoder_lstm(in_2, (h_n, c_n))
            #out = pred.squeeze(-1)


            h_n = torch.zeros(self.hyperparameters["num_layers"]*self.bidir_factor, x.shape[0],  self.hidden_size[0]).to(self.device)
            pred, _ = self.decoder_lstm(in_2, (h_n, c_n))
            
            pred = [self.fc_end(pred[:, i, :]) for i in range(self.y_horizon)]
            pred = torch.transpose(torch.stack(pred), 0, 1).squeeze(-1)
            if "pred_normalized" in self.hyperparameters["info"]:
                out = pred
            else:
                out = pred
        
            #out_dec, _ = self.decoder_lstm(in_2, (h_n, c_n))
            #pred = [self.last_activation(self.fc_end(out_dec[:, i, :])) for i in range(self.y_horizon)]
            #out = torch.transpose(torch.stack(pred), 0, 1).squeeze(-1) 
            
            predicitons.append(out[:, :, None])

            if self.hyperparameters["include_x_features"]:
                x_new = torch.cat((out[:, :, None], y_feat_i), dim=-1)
            else:
                x_new = out[:, None]
            
            x = torch.cat((x, x_new), dim=1)[:, self.y_horizon:]
              
            if "coursenumber" in self.hyperparameters["features"]:
                X_course_emb = torch.cat((X_course_emb, y_emb_i), dim=1)[:,self.y_horizon:]

        if len(predicitons) == 0:
            return torch.tensor([])
        else:
            return torch.cat(predicitons, dim=1).squeeze(0)
                
    def forecast_iter_old(self, x, y_features, len_y, room_id):
        
        x = x[None, :]
        y_features = y_features[None, :]
        
        if "coursenumber" in self.hyperparameters["features"]:
            
            X_course_emb = self.course_embedding(room_id[:self.x_horizon])[None, :]
            y_course_emb = self.course_embedding(room_id[self.x_horizon:])[None, :]
            
            x = torch.cat((x, X_course_emb), -1)
            y_features = torch.cat((y_features, y_course_emb), -1)
        
        if self.hyperparameters["differencing"] == "whole":
            x[:, :, 0] = x[:, :, 0]*0.5 +  0.5
            
        _, (_, c_n) = self.encoder_lstm(x)

        h_n = torch.zeros(self.hyperparameters["num_layers"]*self.bidir_factor, x.shape[0],  self.hidden_size[0]).to(self.device)
        pred, _ = self.decoder_lstm(y_features, (h_n, c_n))
        
        pred = [self.fc_end(pred[:, i, :]) for i in range(len_y)]
        pred = torch.stack(pred).squeeze(-1) 

        return pred







class MassConservingOccLSTM(nn.Module):
    
    def __init__(self, hyperparameters,  **kwargs):
        super().__init__()

        self.hyperparameters = hyperparameters
        
        self.x_size = hyperparameters["x_size"]
        self.x_horizon = hyperparameters["x_horizon"]
        
        self.y_features_size = hyperparameters["y_features_size"]
        self.y_horizon = hyperparameters["y_horizon"]
        self.y_size = hyperparameters["y_size"]
    
        self.hidden_size = hyperparameters["hidden_size"]
        
        self.occcount = hyperparameters["occcount"]
        self.batch_size = hyperparameters["batch_size"]
        
        self.enc_size = 0
        self.bidir_factor = 2 if hyperparameters["bidirectional"] else 1
        
        if "coursenumber" in hyperparameters["features"]:
            
            with open("data/helpers_occpred.json", "r") as f:
                self.helper = json.load(f)       
        
            self.enc_dim= 5
            course_numbers = self.helper["course_numbers"]
            weights = torch.zeros(len(course_numbers), self.enc_dim)
            self.course_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
            
            self.enc_size = self.enc_dim
          
        self.input_size_1 = self.x_size + self.enc_size
        self.input_size_2 = self.y_features_size + self.enc_size
        self.output_size = self.y_size * self.y_horizon
        
        
        # lstm encoder
        self.encoder_mclstm = MassConservingLSTM(
            1,
            self.x_size + self.enc_size -1,
            self.hidden_size[0],
            batch_first=True,
            time_dependent=True
        )
        self.encoder_mclstm.reset_parameters()
        
        # lstm decoder
        self.decoder_lstm = torch.nn.LSTM(
            self.input_size_2, 
            self.hidden_size[0], 
            batch_first=True, 
            bidirectional=hyperparameters["bidirectional"],
            num_layers=hyperparameters["num_layers"]
        )
        
        # predictor
        self.fc_end = nn.Sequential()
        self.fc_end.add_module("input_layer", nn.Linear(self.hidden_size[0]*self.bidir_factor, self.y_size ))
        
        if len(self.hidden_size) > 1:
            raise NotImplementedError("Only one hidden layer is supported")
        
        if self.occcount:
            raise
            self.last_activation = nn.ReLU()    
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.forward_print_count = 0

      
    def forward(self, x, y_features, room_id=None):
    
        if "coursenumber" in self.hyperparameters["features"]:
            
            X_course_emb = self.course_embedding(room_id[:, :self.x_horizon])
            y_course_emb = self.course_embedding(room_id[:, self.x_horizon:])
            
            x = torch.cat((x, X_course_emb), -1)
            y_features = torch.cat((y_features, y_course_emb), -1)
        
        xm = x[:, :, 0][:, :, None]
        xa = x[:, :, 1:]
        
        if self.hyperparameters["differencing"] == "whole":
            xm = xm*0.5 +  0.5
            
        h_n , c_n = self.encoder_mclstm(xm, xa)
        c_n = c_n[:, -1, :][None, :, :].repeat(self.hyperparameters["num_layers"]*self.bidir_factor, 1, 1)

        h_n = torch.zeros(self.hyperparameters["num_layers"]*self.bidir_factor, x.shape[0],  self.hidden_size[0]).to(self.device)

        pred, _ = self.decoder_lstm(y_features, (h_n, c_n))
        
        pred = [self.fc_end(pred[:, i, :]) for i in range(self.y_horizon)]
        pred = torch.transpose(torch.stack(pred), 0, 1).squeeze(-1) 
        
        pred = pred.squeeze(-1)
        
        
        #if self.forward_print_count% 250 == 0:
        #    print(pred.min(), pred.max())
        #self.forward_print_count += 1
        
        return pred    
    
    def forecast_iter(self, x, y_features, len_y, room_id):
        x = x[None, :]
        y_features = y_features[None, :]
        
        if "coursenumber" in self.hyperparameters["features"]:
            X_course_emb = self.course_embedding(room_id[:self.x_horizon])[None, :]
            y_course_emb = self.course_embedding(room_id[self.x_horizon:])[None, :]
    
        predicitons = []
        for i in range(0, len_y, self.y_horizon):
                
            y_feat_i = y_features[:, i:i+self.y_horizon]    
            
            if y_feat_i.shape[1] < self.y_horizon:
                break
            
            if "coursenumber" in self.hyperparameters["features"]:
                
                y_emb_i = y_course_emb[:, i:i+self.y_horizon]
                x_emb_i = X_course_emb

                in_1 = torch.cat((x, x_emb_i), -1)
                in_2 = torch.cat((y_feat_i, y_emb_i), -1)
            
            else:
                raise
                in_1 = torch.cat((x, x_emb_i), -1)
                in_2 = torch.cat((y_feat_i, y_emb_i), -1)
        
            xm = in_1[:, :, 0][:, :, None]
            xa = in_1[:, :, 1:]
            
            if self.hyperparameters["differencing"] == "whole":
                xm = xm*0.5 +  0.5
                
            h_n , c_n = self.encoder_mclstm(xm, xa)
            
            c_n = c_n[:, -1, :][None, :, :].repeat(self.hyperparameters["num_layers"]*self.bidir_factor, 1, 1)
            h_n = torch.zeros(self.hyperparameters["num_layers"]*self.bidir_factor, x.shape[0],  self.hidden_size[0]).to(self.device)
            pred, _ = self.decoder_lstm(in_2, (h_n, c_n))
            
            pred = [self.fc_end(pred[:, i, :]) for i in range(self.y_horizon)]
            pred = torch.transpose(torch.stack(pred), 0, 1).squeeze(-1) 
            
            pred = pred.squeeze(-1)
        
        
            #out_dec, _ = self.decoder_lstm(in_2, (h_n, c_n))
            #pred = [self.last_activation(self.fc_end(out_dec[:, i, :])) for i in range(self.y_horizon)]
            #out = torch.transpose(torch.stack(pred), 0, 1).squeeze(-1) 
            out = pred
            predicitons.append(out[:, :, None])

            if self.hyperparameters["include_x_features"]:
                x_new = torch.cat((out[:, :, None], y_feat_i), dim=-1)
            else:
                x_new = out[:, None]
            
            x = torch.cat((x, x_new), dim=1)[:, self.y_horizon:]
              
            if "coursenumber" in self.hyperparameters["features"]:
                X_course_emb = torch.cat((X_course_emb, y_emb_i), dim=1)[:,self.y_horizon:]

        if len(predicitons) == 0:
            return torch.tensor([])
        else:
            return torch.cat(predicitons, dim=1).squeeze(0)
                    
class MassConservingLSTM(nn.Module):
    """ Pytorch implementation of Mass-Conserving LSTMs. """

    def __init__(self, in_dim: int, aux_dim: int, out_dim: int,
                 in_gate: nn.Module = None, out_gate: nn.Module = None,
                 redistribution: nn.Module = None, time_dependent: bool = True,
                 batch_first: bool = False):
        """
        Parameters
        ----------
        in_dim : int
            The number of mass inputs.
        aux_dim : int
            The number of auxiliary inputs.
        out_dim : int
            The number of cells or, equivalently, outputs.
        in_gate : nn.Module, optional
            A module computing the (normalised!) input gate.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `in_dim` x `out_dim` matrix for every sample.
            Defaults to a time-dependent softmax input gate.
        out_gate : nn.Module, optional
            A module computing the output gate.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `out_dim` vector for every sample.
        redistribution : nn.Module, optional
            A module computing the redistribution matrix.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `out_dim` x `out_dim` matrix for every sample.
        time_dependent : bool, optional
            Use time-dependent gates if `True` (default).
            Otherwise, use only auxiliary inputs for gates.
        batch_first : bool, optional
            Expects first dimension to represent samples if `True`,
            Otherwise, first dimension is expected to represent timesteps (default).
        """
        super().__init__()
        self.in_dim = in_dim
        self.aux_dim = aux_dim
        self.out_dim = out_dim
        self._seq_dim = 1 if batch_first else 0

        gate_kwargs = {
            'aux_dim': aux_dim,
            'out_dim': out_dim if time_dependent else None,
            'normaliser': nn.Softmax(dim=-1),
        }
        if redistribution is None:
            redistribution = MCGate((out_dim, out_dim), **gate_kwargs)
        if in_gate is None:
            in_gate = MCGate((in_dim, out_dim), **gate_kwargs)
        if out_gate is None:
            gate_kwargs['normaliser'] = nn.Sigmoid()
            out_gate = MCGate((out_dim, ), **gate_kwargs)

        self.redistribution = redistribution
        self.in_gate = in_gate
        self.out_gate = out_gate

    @property
    def batch_first(self) -> bool:
        return self._seq_dim != 0

    def reset_parameters(self, out_bias: float = -3.):
        """
        Parameters
        ----------
        out_bias : float, optional
            The initial bias value for the output gate (default to -3).
        """
        self.redistribution.reset_parameters(bias_init=nn.init.eye_)
        self.in_gate.reset_parameters(bias_init=nn.init.zeros_)
        self.out_gate.reset_parameters(
            bias_init=lambda b: nn.init.constant_(b, val=out_bias)
        )

    def forward(self, xm, xa, state=None):
        xm = xm.unbind(dim=self._seq_dim)
        xa = xa.unbind(dim=self._seq_dim)

        if state is None:
            state = self.init_state(len(xa[0]))

        hs, cs = [], []
        for xm_t, xa_t in zip(xm, xa):
            h, state = self._step(xm_t, xa_t, state)
            hs.append(h)
            cs.append(state)

        hs = torch.stack(hs, dim=self._seq_dim)
        cs = torch.stack(cs, dim=self._seq_dim)
        return hs, cs

    @torch.no_grad()
    def init_state(self, batch_size: int):
        """ Create the default initial state. """
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.out_dim, device=device)

    def _step(self, xm_t, xa_t, c_t):
        """ Implementation of MC-LSTM recurrence. """
        r = self.redistribution(xm_t, xa_t, c_t)
        i = self.in_gate(xm_t, xa_t, c_t)
        o = self.out_gate(xm_t, xa_t, c_t)

        c = torch.matmul(c_t.unsqueeze(-2), r).squeeze(-2)
        c = c + torch.matmul(xm_t.unsqueeze(-2), i).squeeze(-2)
        h = o * c
        c = c - h
        return h, c

    def autoregress(self, c0: torch.Tensor, xa: torch.Tensor, xm: torch.Tensor = None):
        """
        Use MC-LSTM in an autoregressive fashion.

        By operating on the cell states of MC-LSTM directly,
        the MC-LSTM can be used as an auto-regressive model.

        Parameters
        ----------
        c0 : (B, out_dim) torch.Tensor
            The initial cell state for the MC-LSTM or, equivalently,
            the starting point for the auto-regression.
        xa : (L, B, aux_dim) torch.Tensor
            A sequence of auxiliary inputs for the MC-LSTM.
            The output sequence will have the same length `L` as the given sequence.
            If not specified, the sequence consists of
            `length` equally spaced points between 0 and 1.
        xm : (L, B, in_dim) torch.Tensor, optional
            A sequence of mass inputs for the MC-LSTM.
            This sequence must have the same length as `xa`.
            If not specified, a sequence of zeros is used.

        Returns
        -------
        y : (L, B, out_dim) torch.Tensor
            The sequence of cell states from the MC-LSTM or, equivalently,
            the outputs of the autoregression.
            The length of the sequence is specified is the length of `xa`.
        h : (L, B, out_dim) torch.Tensor
            The sequence of outputs produced by the MC-LSTM.
            This sequence should contain all mass that disappeared over time,
            and has the same length as `y`.

        Notes
        -----
        If `self.batch_first` is `True`, the documented shapes of
        input and output sequences should be `(B, L, ...)` instead of `(L, B, ...)`.

        """
        if len(c0.shape) != 2 or c0.size(1) != self.out_dim:
            raise ValueError(f"cell state must have shape (?, {self.out_dim})")
        if xa.size(-1) != self.aux_dim:
            raise ValueError(f"auxiliary input must have shape (..., {self.aux_dim})")
        if xm is None:
            xm = torch.zeros(*xa.shape[:-1], self.in_dim)
        elif xm.size(-1) != self.in_dim:
            raise ValueError(f"mass input must have shape (..., {self.in_dim})")

        h, y = self.forward(xm, xa, state=c0)
        return y, h

class MCGate(nn.Module):
    """ Default gating logic for MC-LSTM. """

    def __init__(self, shape: tuple, aux_dim: int, out_dim: int = None,
                 in_dim: int = None, normaliser: nn.Module = nn.Softmax(dim=-1)):
        """
        Parameters
        ----------
        shape : tuple of ints
            The output shape for this gate.
        aux_dim : int
            The number of auxiliary inputs per timestep.
        out_dim : int, optional
            The number of accumulation cells.
            If `None`, the cell states are not used in the gating (default).
        in_dim : int, optional
            The number of mass inputs per timestep.
            If `None`, the mass inputs are not used in the gating (default).
        normaliser : nn.Module, optional
            The activation function to use for computing the gates.
            This function is responsible for any normalisation of the gate.
        """
        super().__init__()
        batch_dim = 1 if any(n == 0 for n in shape) else -1
        self.out_shape = (batch_dim, *shape)
        self.use_mass = in_dim is not None
        self.use_state = out_dim is not None

        gate_dim = aux_dim
        if self.use_mass:
            gate_dim += in_dim
        if self.use_state:
            gate_dim += out_dim

        self.connections = nn.Linear(gate_dim, np.prod(shape).item())
        self.normaliser = normaliser

    def reset_parameters(self, bias_init=nn.init.zeros_):
        """
        Parameters
        ----------
        bias_init : callable
            Initialisation function for the bias parameter (in-place).
        """
        bias_init(self.connections.bias.view(self.out_shape[1:]))
        nn.init.orthogonal_(self.connections.weight)

    def forward(self, xm, xa, c):
        inputs = [xa]
        if self.use_mass:
            xm_sum = torch.sum(xm, dim=-1, keepdims=True)
            scale = torch.where(xm_sum == 0, xm_sum.new_ones(1), xm_sum)
            inputs.append(xm / scale)
        if self.use_state:
            c_sum = torch.sum(c, dim=-1, keepdims=True)
            scale = torch.where(c_sum == 0, c_sum.new_ones(1), c_sum)
            inputs.append(c / scale)

        x_ = torch.cat(inputs, dim=-1)
        s = self.connections(x_)
        s = s.view(self.out_shape)
        return self.normaliser(s)


import torch.functional as F
import math 
from torch.nn import LSTM

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        hx, cx = hidden
        gates = self.x2h(x) + self.h2h(hx)
    
        gates = gates.squeeze()
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)        

        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        

        hy = torch.mul(outgate, F.tanh(cy))
        
        return (hy, cy)

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        cell_list=[]
        
        cell_list.append(LSTMCell( self.input_size, self.hidden_size))#the first
        #one has a different number of input channels
        
        for idcell in range(1,self.num_layers):
            cell_list.append(LSTMCell(self.hidden_size, self.hidden_size))
        self.cell_list=nn.ModuleList(cell_list)      
    
    def forward(self, current_input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W
        """
        #current_input=input
        next_hidden=[]#hidden states(h and c)
        seq_len=current_input.size(0)

        
        for idlayer in range(self.num_layers):#loop for every layer

            hidden_c=hidden_state[idlayer]#hidden and c are images with several channels
            all_output = []
            output_inner = []            
            for t in range(seq_len):#loop for every step
                hidden_c=self.cell_list[idlayer](current_input,hidden_c)

                output_inner.append(hidden_c)

            next_hidden.append(hidden_c)
            current_input = hidden_c[0]
    
        return next_hidden


class SimpleOccGRU(torch.nn.Module):
    
    def __init__(self, hyperparameters, **kwargs):
        
        super().__init__()
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.hyperparameters = hyperparameters

        self.x_size = hyperparameters["x_size"]
        self.x_horizon = hyperparameters["x_horizon"]
        
        self.y_horizon = hyperparameters["y_horizon"]
        self.y_features_size = hyperparameters["y_features_size"]
        self.y_size = hyperparameters["y_size"]
        
        self.output_size = hyperparameters["y_size"] * hyperparameters["y_horizon"]
    
        self.hidden_size = hyperparameters["hidden_size"]
        
        self.batch_size = hyperparameters["batch_size"]
        ############ Model Definition ############
        
        self.enc_size = 0
        
        if "coursenumber" in hyperparameters["features"]:
            
            with open("data/helpers_occpred.json", "r") as f:
                self.helper = json.load(f)       
        
            self.enc_dim= 5
            course_numbers = self.helper["course_numbers"]
            weights = torch.zeros(len(course_numbers), self.enc_dim)
            self.course_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
            
            self.enc_size = self.enc_dim
          
        self.input_size = self.x_size + self.enc_size
        
        # lstm layer
        self.gru = torch.nn.GRU(self.input_size,  self.hidden_size[0], batch_first=True, num_layers=hyperparameters["num_layers"])
          
        # fc at end
        self.fc_end = nn.Sequential()
        if len(self.hidden_size) == 2:
            
            self.fc_end.add_module("input_layer", nn.Linear(self.hidden_size[0], self.hidden_size[1]))
            
            if hyperparameters["layer_norm"]:
                self.fc_end.add_module("layer_norm", nn.LayerNorm(self.hidden_size[1]))
                
            self.fc_end.add_module("relu_0", nn.ReLU())
            self.fc_end.add_module("output_layer", nn.Linear(self.hidden_size[1], self.y_size))
        
        else:
            self.fc_end.add_module("input_layer", nn.Linear(self.hidden_size[0], self.y_size))
            
        if "forget_gate" in hyperparameters:
            if not hyperparameters["forget_gate"]:
                for name, param in self.lstm.named_parameters():
                    if 'bias_ih' in name or 'bias_hh' in name:
                        bias_size = param.size(0) // 4 
                        param.data[bias_size:2*bias_size].fill_(25)
                    
        # last activation function
        if hyperparameters["occcount"]:
            self.last_activation = nn.ReLU()    
        else:
            self.last_activation = nn.Identity()
            
        ## freeze h_0 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def forward(self, x, y_features, room_id=None):
        
        if "coursenumber" in self.hyperparameters["features"]:
            
            X_course_emb = self.course_embedding(room_id[:, :self.x_horizon])
            y_course_emb = self.course_embedding(room_id[:, self.x_horizon:])
            
            x = torch.cat((x, X_course_emb), -1)
            y_features = torch.cat((y_features, y_course_emb), -1)

        out, h_n = self.gru(x)       
        
        y_t = self.last_activation(self.fc_end(out[:, -1, :]))  
        y_t = y_t[:, None, :]
        
        pred_list = []
        for i in range(0, self.hyperparameters["y_horizon"]):
            
            y_feat_t = y_features[:, i, :]
            y_in = torch.cat((y_t, y_feat_t[:, None, :]), 2)
            
            h_t1, h_n = self.gru(y_in, h_n)
            y_t = self.last_activation(self.fc_end(h_t1))

            pred_list.append(y_t)
        
        pred_list = torch.cat(pred_list, dim=1).squeeze(-1)
        
        return pred_list
    
    def forecast_iter(self, x, y_features, len_y, room_id):
        
        x = x[None, :]
        y_features = y_features[None, :]
        
        if "coursenumber" in self.hyperparameters["features"]:
            
            X_course_emb = self.course_embedding(room_id[:self.x_horizon])[None, :]
            y_course_emb = self.course_embedding(room_id[self.x_horizon:])[None, :]
            
            x = torch.cat((x, X_course_emb), -1)
            y_features = torch.cat((y_features, y_course_emb), -1)


        out, h_n = self.gru(x)     
        y_t = self.last_activation(self.fc_end(out[:, -1, :]))[:, None, :]
        
        
        pred_list = []
        for i in range(0, len_y):
            
            y_feat_i = y_features[:, i, :][None, :]
            y_t_in = torch.cat((y_t, y_feat_i), 2)
        
            out, h_n = self.gru(y_t_in, h_n)

            y_t = self.last_activation(self.fc_end(out))

            pred_list.append(y_t.squeeze(-1))
        
        return torch.cat(pred_list)

class SimpleOccTransformer(torch.nn.Module):
    
    def __init__(self, hyperparameters, **kwargs):
        
        super().__init__()
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.hyperparameters = hyperparameters

        self.x_size = hyperparameters["x_size"]
        self.x_horizon = hyperparameters["x_horizon"]
        
        self.y_horizon = hyperparameters["y_horizon"]
        self.y_features_size = hyperparameters["y_features_size"]
        self.y_size = hyperparameters["y_size"]
        
        self.output_size = hyperparameters["y_size"] * hyperparameters["y_horizon"]
    
        self.hidden_size = hyperparameters["hidden_size"]
        
        self.batch_size = hyperparameters["batch_size"]
        ############ Model Definition ############
        
        self.enc_size = 0
        
        if "coursenumber" in hyperparameters["features"]:
            
            with open("data/helpers_occpred.json", "r") as f:
                self.helper = json.load(f)       
        
            self.enc_dim= 5
            course_numbers = self.helper["course_numbers"]
            weights = torch.zeros(len(course_numbers), self.enc_dim)
            self.course_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
            
            self.enc_size = self.enc_dim
          
        self.input_size = self.x_size + self.enc_size

        self.transformer_x = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.input_size, 
                nhead=1, 
                dim_feedforward=self.input_size
            ),
            num_layers=hyperparameters["num_layers"]
        )
        
        self.transformer_out = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.y_features_size+self.enc_size, 
                nhead=1, 
                dim_feedforward=self.y_features_size+self.enc_size
            ),
            num_layers=hyperparameters["num_layers"]
        )
        
        
        
        # lstm layer
        #self.lstm = torch.nn.LSTM(self.input_size,  self.hidden_size[0], batch_first=True, num_layers=hyperparameters["num_layers"])
          
        # fc at end
        self.fc_end = nn.Sequential()
        if len(self.hidden_size) == 2:
            
            self.fc_end.add_module("input_layer", nn.Linear(self.hidden_size[0], self.hidden_size[1]))
            
            if hyperparameters["layer_norm"]:
                self.fc_end.add_module("layer_norm", nn.LayerNorm(self.hidden_size[1]))
                
            self.fc_end.add_module("relu_0", nn.ReLU())
            self.fc_end.add_module("output_layer", nn.Linear(self.hidden_size[1], self.y_size))
        
        else:
            self.fc_end.add_module("input_layer", nn.Linear(self.hidden_size[0], self.y_size))
            
        if "forget_gate" in hyperparameters:
            if not hyperparameters["forget_gate"]:
                for name, param in self.lstm.named_parameters():
                    if 'bias_ih' in name or 'bias_hh' in name:
                        bias_size = param.size(0) // 4 
                        param.data[bias_size:2*bias_size].fill_(25)
                    
        # last activation function
        if hyperparameters["occcount"]:
            self.last_activation = nn.ReLU()    
        else:
            self.last_activation = nn.Identity()
            
        ## freeze h_0 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def forward(self, x, y_features, room_id=None):
    
        if "coursenumber" in self.hyperparameters["features"]:
            
            X_course_emb = self.course_embedding(room_id[:, :self.x_horizon])
            y_course_emb = self.course_embedding(room_id[:, self.x_horizon:])
            
            x = torch.cat((x, X_course_emb), -1)
            y_features = torch.cat((y_features, y_course_emb), -1)
        
        out_x = self.transformer_in(x)
        out = self.transformer_out(y_features)
        #print(out_x.shape, out_y.shape)
        pred = self.last_activation(self.fc_end(out[:, -1, :]))

        raise
        return pred
    
    def forecast_iter(self, x, y_features, len_y, room_id):
        
        x = x[None, :]
        y_features = y_features[None, :]
        
        if "coursenumber" in self.hyperparameters["features"]:
            X_course_emb = self.course_embedding(room_id[:self.x_horizon])[None, :]
            y_course_emb = self.course_embedding(room_id[self.x_horizon:])[None, :]
    
        predicitons = []
        for i in range(0, len_y, self.y_horizon):

            y_feat_i = y_features[:, i:i+self.y_horizon]    
            
            if y_feat_i.shape[1] < self.y_horizon:
                break
            
            if "coursenumber" in self.hyperparameters["features"]:
                
                y_emb_i = y_course_emb[:, i:i+self.y_horizon]
                x_emb_i = X_course_emb
            
                in_1 = torch.cat((x, x_emb_i), -1)
                in_2 = torch.cat((y_feat_i, y_emb_i), -1)
            
            else:

                in_1 = torch.cat((x, x_emb_i), -1)
                in_2 = torch.cat((y_feat_i, y_emb_i), -1)
            
            _, (h_n, c_n) = self.encoder_lstm(in_1)

            out_dec, _ = self.decoder_lstm(in_2, (h_n, c_n))

            out = self.last_activation(self.fc_end(out_dec[:, -1, :]))
            
            predicitons.append(out[:, :, None])

            if self.hyperparameters["include_x_features"]:
                x_new = torch.cat((out[:, :, None], y_feat_i), dim=-1)
            else:
                x_new = out[:, None]
            
            x = torch.cat((x, x_new), dim=1)[:, self.y_horizon:]
              
            if "coursenumber" in self.hyperparameters["features"]:
                X_course_emb = torch.cat((X_course_emb, y_emb_i), dim=1)[:,self.y_horizon:]

        if len(predicitons) == 0:
            return torch.tensor([])
        else:
            return torch.cat(predicitons)
        

#class EncDecOccLSTM(torch.nn.Module):
    
#    def __init__(self, hyperparameters, **kwargs):
        
#        super().__init__()
        
#        self.hyperparameters = hyperparameters

#        self.x_size = hyperparameters["x_size"]
#        self.y_features_size = hyperparameters["y_features_size"]
#        #self.input_size = self.x_size + self.y_features_size 
    
#        self.hidden_size = hyperparameters["hidden_size"]
        
#        self.occcount = hyperparameters["occcount"]
#        self.batch_size = hyperparameters["batch_size"]
        
#        self.output_size = hyperparameters["y_size"] * hyperparameters["y_horizon"]
        
#        # lstm encoder
#        self.encoder_lstm = torch.nn.LSTM(self.x_size, self.hidden_size[0], batch_first=True)
#        # lstm decoder
#        self.decoder_lstm = torch.nn.LSTM(self.y_features_size, self.hidden_size[0], batch_first=True)
        
#        # predictor
#        self.linear = torch.nn.Linear(self.hidden_size[0], hyperparameters["y_size"])
        
#        if self.occcount:
#            self.last_activation = nn.ReLU()    
#        else:
#            self.last_activation = nn.Sigmoid()
            
#        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
#    def forward(self, x, y_features):
        
#        _, (_, c_n) = self.encoder_lstm(x)
        
#        out, _ = self.decoder_lstm(y_features, (torch.zeros(1, self.batch_size, self.hidden_size[0]).to(self.device), c_n))
        
#        pred_list = [ self.last_activation(self.linear(out[:, i, :])) for i in range(self.hyperparameters["y_horizon"])]
                
#        return torch.stack(pred_list, 1).squeeze()
    
    
    
################## Run 3 #########################

# works fine !!!!!!!!1RUN2!!!!!!
#class EncDecOccLSTM1(torch.nn.Module):
    
#    def __init__(self, hyperparameters, **kwargs):
        
#        super().__init__()
        
#        self.hyperparameters = hyperparameters

#        self.x_size = hyperparameters["x_size"]
#        self.y_features_size = hyperparameters["y_features_size"]
#        #self.input_size = self.x_size + self.y_features_size 
    
#        self.hidden_size = hyperparameters["hidden_size"]
        
#        self.occcount = hyperparameters["occcount"]
#        self.batch_size = hyperparameters["batch_size"]
        
#        self.output_size = hyperparameters["y_size"] * hyperparameters["y_horizon"]
        
#        # lstm encoder
#        self.encoder_lstm = torch.nn.LSTM(self.x_size, self.hidden_size[0], batch_first=True, num_layers=hyperparameters["num_layers"])
#        # lstm decoder
#        self.decoder_lstm = torch.nn.LSTM(self.y_features_size, self.hidden_size[0], proj_size=1, batch_first=True, num_layers=hyperparameters["num_layers"])
        
#        #print(self.decoder_lstm)
#        #raise ValueError("Stop")
#        if self.occcount:
#            self.last_activation = nn.ReLU()    
#        else:
#            self.last_activation = nn.Sigmoid()
            
#        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
#    def forward(self, x, y_features):
        
#        _, (_, c_n) = self.encoder_lstm(x)
        
#        out, (_, _) = self.decoder_lstm(
#            y_features, (
#                torch.zeros(self.hyperparameters["num_layers"], self.batch_size, 1).to(self.device), 
#                c_n
#                )
#            )

#       return out.squeeze()
    
###### Does not work ##########
#class EncDecOccLSTM1(torch.nn.Module):
#     #"""Sucks!!!!!!!!!!!!!!!!"""
#    def __init__(self, hyperparameters, **kwargs):
        
#        super().__init__()
        
#        self.hyperparameters = hyperparameters

#        self.x_size = hyperparameters["x_size"]
#        self.y_features_size = hyperparameters["y_features_size"]
#        #self.input_size = self.x_size + self.y_features_size 
    
#        self.hidden_size = hyperparameters["hidden_size"]
        
#        self.occcount = hyperparameters["occcount"]
#        self.batch_size = hyperparameters["batch_size"]
        
#        self.output_size = hyperparameters["y_size"] * hyperparameters["y_horizon"]
        
#        # lstm encoder
#        self.encoder_lstm = torch.nn.LSTM(self.x_size, self.hidden_size[0], batch_first=True)
#        # lstm decoder
#        self.decoder_lstm = torch.nn.LSTM(self.y_features_size, self.hidden_size[0], batch_first=True)
#        # predictor
#        self.linear = torch.nn.Linear(self.hidden_size[0] * hyperparameters["y_horizon"], hyperparameters["y_size"])
        
#        if self.occcount:
#            self.last_activation = nn.ReLU()    
#        else:
#            self.last_activation = nn.Sigmoid()
            
#        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
#    def forward(self, x, y_features):
        
#        _, (_, c_n) = self.encoder_lstm(x)
        
#        out, _ = self.decoder_lstm(y_features, (torch.zeros(1, self.batch_size, self.hidden_size[0]).to(self.device), c_n))

#        out = out.reshape(-1, self.hidden_size[0]*self.hyperparameters["y_horizon"])
#        pred = self.last_activation(self.linear(out))
#        return pred
#        #pred_list = [ self.last_activation(self.linear(out[:, i, :])) for i in range(self.hyperparameters["y_horizon"])]
#        #return orch.stack(pred_list, 1).squeeze()
