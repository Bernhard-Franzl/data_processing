import torch
import torch.nn as nn
import json
class SimpleLectureDenseNet(nn.Module):

    def __init__(self, hyperparameters):
        super().__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.hyperparameters = hyperparameters

        self.x_size = hyperparameters["x_size"] 
        self.y_features_size = hyperparameters["y_features_size"] 
        self.output_size = hyperparameters["y_size"]
    
        self.hidden_size = hyperparameters["hidden_size"]
        
        self.occcount = hyperparameters["occcount"]
        self.batch_size = hyperparameters["batch_size"]
        
        self.immutable_size = hyperparameters["immutable_size"]
        self.dropout_p = hyperparameters["dropout"]
        
        self.emb_dim = hyperparameters["embedding_dim"]
        self.enc_size = 0
        if "coursenumber" in hyperparameters["features"]:
            with open("data/helpers.json", "r") as f:
                self.helper = json.load(f)
                
            self.course_numbers = self.helper["course_numbers"]
            
            weights = torch.zeros(len(self.course_numbers), self.emb_dim)
            self.course_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
            
            self.enc_size = self.emb_dim-1
            
        
        #self.lin_1 = nn.Linear(self.x_size, self.hidden_size[0])
        #self.lin_2 = nn.Linear(self.y_features_size, self.hidden_size[0])
       # self.lin_3 = nn.Linear(self.immutable_size-1+10, self.hidden_size[0])
        
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.lin_mid = nn.Linear(self.x_size + self.y_features_size + self.immutable_size + self.enc_size, self.hidden_size[1])
        
        self.batch_norm = nn.BatchNorm1d(self.hidden_size[1])
        
        self.lin_out = nn.Linear(self.hidden_size[1], self.output_size)
        
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.discretization = hyperparameters["discretization"]
        
        
        if self.discretization:
            self.last_activation = nn.Identity()
        elif self.occcount:
            self.last_activation = nn.ReLU()
        else:
            self.last_activation = nn.Identity()
            
        #self.room_encoding = torch.eye(2).to(self.device)
        #self.room_enc_size = self.room_encoding.shape[1]
        
        #self.input_size = self.x_size + self.y_features_size + self.immutable_size
        #self.model = nn.Sequential()
        #if len(self.hidden_size) > 1:
        #    self.model.add_module("input_layer", nn.Linear(self.input_size, self.hidden_size[0]))
        #    self.model.add_module("relu_0", nn.LeakyReLU())
            
        #    for i in range(0, len(self.hidden_size)-1):
        #        self.model.add_module(f"hidden_layer_{i}", nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))
        #        self.model.add_module(f"relu_{i+1}", nn.LeakyReLU())
            
        #    self.model.add_module("output_layer", nn.Linear(self.hidden_size[-1], self.output_size))  
        #else:
        #    if len(self.hidden_size) == 0:
        #        self.model.add_module("input_layer", nn.Linear(self.input_size, self.output_size))
        #    else:
        #        self.model.add_module("input_layer", nn.Linear(self.input_size, self.hidden_size[0]))
        #        self.model.add_module("relu_0", nn.LeakyReLU())
        #        self.model.add_module("output_layer", nn.Linear(self.hidden_size[0], self.output_size))
            
        #if self.occcount:
        #    if self.hyperparameters["differencing"] in ["sample", "whole"]:
        #        raise ValueError("Not implemented")
        #    else:
        #        self.model.add_module("acti_out", nn.ReLU())

        #else:
            #if self.hyperparameters["differencing"] in ["sample", "whole"]:
            #    raise ValueError("Not implemented")
            #    self.model.add_module("acti_out", nn.Tanh())
            #    #print("Tanh added")
            #else:
            #    if self.hyperparameters["criterion"] == "CE":
            #        print("nothing added")
            #    else:   
            #        self.model.add_module("acti_out", nn.Sigmoid())
   
    def forward(self, x, y_features, room_id=None):
        
        #room_enc = self.room_encoding[room_id]
        #room_enc = room_enc.view(self.batch_size, -1)

        # room_id is actually immutable features array
        
        
        if "coursenumber" in self.hyperparameters["features"]:
            course_id = room_id[:, -1].to(torch.int64)
            course_emb = self.course_embedding(course_id)
            immutable_features = torch.cat((room_id[:, :-1], course_emb), 1)
            
        else:
            immutable_features = room_id.view(self.batch_size, self.immutable_size)
        
        x = x.view(self.batch_size, self.x_size)
        y_features = y_features.view(self.batch_size, self.y_features_size)
        
        #manual model generation
        input_cat = torch.cat((x, y_features, immutable_features), 1)
        
        #in_mid = self.dropout(in_mid)
        out_mid = self.lin_mid(input_cat)
        
        #out_mid = self.batch_norm(out_mid)
        out_mid = self.relu(out_mid)
        #out_mid = self.dropout(out_mid)
        
        pred = self.lin_out(out_mid)
        #pred = self.last_activation(out)
        
        # advanced model generation
        #input = torch.cat((x, y_features, immutable_features), 1)
        #out = self.model(input)
    
        return pred
    
    def forecast_iter(self, x, y_features, len_y, room_id=None):
        
        room_enc = self.room_encoding[room_id]

        predicitons = []
        
        for i in range(0, len_y, self.y_horizon):
            
            room_enc_flattened = room_enc.view(-1)
            x_flattened  = x.view(-1)
            y_feat_i = y_features[i:i+self.y_horizon]
            
            if y_feat_i.shape[0] < self.y_horizon:
                break
            
            y_feat_i_flattend = y_feat_i.reshape(-1)
            
            input = torch.cat((x_flattened, y_feat_i_flattend, room_enc_flattened))
            
            out = self.model(input)

                
            predicitons.append(out[:, None])

            if self.hyperparameters["include_x_features"]:
                x_new = torch.cat((out[:, None], y_feat_i), dim=1)
            else:
                x_new = out[:, None]
                
            x = torch.cat((x, x_new), dim=0)[self.y_horizon:]
        
        if len(predicitons) == 0:
            return torch.tensor([])
        else:
            return torch.cat(predicitons)

class SimpleLectureLSTM(torch.nn.Module):
    
    def __init__(self, hyperparameters, **kwargs):
        
        super().__init__()
        self.hyperparameters = hyperparameters

        self.x_size = hyperparameters["x_size"]
        self.y_features_size = hyperparameters["y_features_size"]
    
        self.output_size = hyperparameters["y_size"]
    
        self.hidden_size = hyperparameters["hidden_size"]
        
        self.batch_size = hyperparameters["batch_size"]
        self.immutable_size = hyperparameters["immutable_size"]
        self.discretization = hyperparameters["discretization"]
        
        ############ Model Definition ############
           
        if "coursenumber" in hyperparameters["features"]:
            
            self.emb_dim = hyperparameters["embedding_dim"]
            
            with open("data/helpers_lecture_random_2.json", "r") as f:
                self.helper = json.load(f)
                
            self.course_numbers = self.helper["course_numbers"]
            
            weights = torch.zeros(len(self.course_numbers), self.hyperparameters["num_layers"]*self.emb_dim)
            self.course_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
            
            self.linear_emb = torch.nn.Linear(self.hyperparameters["num_layers"]*self.emb_dim, self.hyperparameters["num_layers"]*self.hidden_size[0])
        
        # lstm layer
        self.lstm = torch.nn.LSTM(self.x_size, self.hidden_size[0], batch_first=True, num_layers=hyperparameters["num_layers"])
        
        # fc at end
        self.fc_end = nn.Sequential()
        if len(self.hidden_size) == 2:
            
            self.fc_end.add_module("input_layer", nn.Linear(self.hidden_size[0]+ self.y_features_size + self.immutable_size -1, self.hidden_size[1]))
            
            if hyperparameters["layer_norm"]:
                self.fc_end.add_module("layer_norm", nn.LayerNorm(self.hidden_size[1]))
                
            self.fc_end.add_module("relu_0", nn.ReLU())
            self.fc_end.add_module("output_layer", nn.Linear(self.hidden_size[1], self.output_size))
        
        else:
            self.fc_end.add_module("input_layer", nn.Linear(self.hidden_size[0]+ self.y_features_size + self.immutable_size, self.output_size))
                    
        # last activation function
        if hyperparameters["occcount"]:
            self.last_activation = nn.ReLU()    
        elif self.discretization:
            self.last_activation = nn.Identity()
        else:
            self.last_activation = nn.Identity()
            
        ## freeze h_0 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def forward(self, x, y_features, room_id=None):
        
        x_shape_0, x_shape_1 = x.shape[0], x.shape[1]
        
        input_tensor = x#[None, :]
        y_features = y_features#[None, :]
        
        if "coursenumber" in self.hyperparameters["features"]:
            course_id = room_id[:, -1].to(torch.int64)
            course_emb = self.course_embedding(course_id)
            c_0 = course_emb.view(-1, x_shape_0, self.hidden_size[0])
            
            room_id = room_id[:, :-1]
            
        else:
            c_0 = torch.zeros(self.hyperparameters["num_layers"], x_shape_0, self.hidden_size[0]).to(self.device)
        
        
        h_0 = torch.zeros(self.hyperparameters["num_layers"], x_shape_0, self.hidden_size[0]).to(self.device)
        

        out_lstm, (_, _) = self.lstm(input_tensor, (h_0, c_0))
        
        in_lin = torch.cat((out_lstm[:, -1, :], y_features[:, -1, :], room_id), 1)

        out_lin = self.fc_end(in_lin)
        pred = self.last_activation(out_lin)

        return pred
        
        
        # with -1
        #h_0 = torch.zeros(self.hyperparameters["num_layers"], self.batch_size, self.hidden_size[0]).to(self.device)
        #c_0 = self.linear_in(room_id).repeat(self.hyperparameters["num_layers"], 1, 1)
        #out, (h_n, c_n) = self.lstm(x, (h_0, c_0))    
        #pred = self.last_activation(self.linear_final(out[:, -1, :])) 
        
        #without -1 
        #h_0 = torch.zeros(self.hyperparameters["num_layers"], self.batch_size, self.hidden_size[0]).to(self.device)
        #c_0 = self.linear_in(room_id).repeat(self.hyperparameters["num_layers"], 1, 1)
        #out_1, (h_n, c_n) = self.lstm(x)    
        #y_t = self.last_activation(self.linear_final(out_1[:, -1, :]))[:, None, :] 
        
        #y_in = torch.cat((y_t, y_features), 2)
        #out_2, _ = self.lstm(y_in, (h_n, c_n))
        #pred = self.last_activation(self.linear_final(out_2[:, -1, :]))
        
        #h_0 = torch.zeros(self.hyperparameters["num_layers"], self.batch_size, self.hidden_size[0]).to(self.device)
        #c_0 = self.linear_in(room_id).repeat(self.hyperparameters["num_layers"], 1, 1)
        
        
        #  New sequential model
        #print(x.shape, y_features.shape, room_id.shape)
        #x = x.view(-1, self.x_size)
        #y_features = y_features.view(-1, self.y_features_size)
        #immutable_features = room_id.view(-1, self.immutable_size)
        
        #manual model generation
        out_1 = self.relu(self.lin_1(x))
        out_2 = self.relu(self.lin_2(y_features))
        out_3 = self.relu(self.lin_3(immutable_features))
        
        in_mid= torch.cat((out_1, out_2, out_3), 1)
        out_mid = self.relu(self.lin_mid(in_mid))
        
        pred = self.lin_out(out_mid)
        
        #y_t = y_t[:, None, :]
        
        #y_feat_t = y_features[:, 0, :]
        #y_in = torch.cat((y_t, y_feat_t[:, None, :]), 2)
        
        #h_t1, (h_n, c_n) = self.lstm(y_in, (h_n, c_n))
        #y_t = self.last_activation(self.linear_final(h_t1))
        
        #pred_list = []
        ##for i in range(0, self.hyperparameters["y_horizon"]):
            
        #y_feat_t = y_features[:, i, :]
        #y_in = torch.cat((y_t, y_feat_t[:, None, :]), 2)
        
        #h_t1, (h_n, c_n) = self.lstm(y_in, (h_n, c_n))
        #y_t = self.last_activation(self.linear_final(h_t1))

        #pred_list.append(y_t)
        
        #pred_list = torch.cat(pred_list, dim=1).squeeze(-1)
        
        return pred_list
    
    def forecast_iter(self, x, y_features, len_y, room_id):
        
        x = x[None, :]
        y_features = y_features[None, :]
        
        room_enc = self.room_embedding(room_id)
        room_enc = room_enc.repeat(self.hyperparameters["num_layers"], 1, 1)

        h_0 = torch.zeros(self.hyperparameters["num_layers"], 1, self.hidden_size[0]).to(self.device)
        
        out, (h_n, c_n) = self.lstm(x, (h_0, room_enc))       
    
        y_t = self.last_activation(self.linear_final(out[:, -1, :]))[:, None, :]
        pred_list = []
        for i in range(0, len_y):
            
            y_feat_i = y_features[:, i, :][None, :]
            y_t_in = torch.cat((y_t, y_feat_i), 2)
            
            out, (h_n, c_n) = self.lstm(y_t_in, (h_n, c_n))

            y_t = self.last_activation(self.linear_final(out))
            #print("lstm:", y_t.shape)

            pred_list.append(y_t.squeeze(-1))
        
        return torch.cat(pred_list)    
    
    
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
        self.lstm = torch.nn.LSTM(self.input_size,  self.hidden_size[0], batch_first=True, num_layers=hyperparameters["num_layers"])
          
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
        y_t = self.last_activation(self.linear_final(out[:, -1, :]))[:, None, :]
        
        
        pred_list = []
        for i in range(0, len_y):
            
            y_feat_i = y_features[:, i, :][None, :]
            y_t_in = torch.cat((y_t, y_feat_i), 2)
            
            out, (h_n, c_n) = self.lstm(y_t_in, (h_n, c_n))

            y_t = self.last_activation(self.linear_final(out))

            pred_list.append(y_t.squeeze(-1))
        
        return torch.cat(pred_list)

class EncDecOccLSTM(torch.nn.Module):
    
    def __init__(self, hyperparameters, **kwargs):
        
        super().__init__()
        
        self.hyperparameters = hyperparameters
        
        self.x_size = hyperparameters["x_size"]
        self.y_features_size = hyperparameters["y_features_size"]
    
        self.output_size = hyperparameters["y_size"] * hyperparameters["y_horizon"]
    
        self.hidden_size = hyperparameters["hidden_size"]
        
        self.occcount = hyperparameters["occcount"]
        self.batch_size = hyperparameters["batch_size"]
        
        #self.room_encoding = torch.eye(2).to(self.device)
        #self.room_enc_size = self.room_encoding.shape[1]    
        
        # embedding for hidden states with 0 init
        weights = torch.randn(2, self.hidden_size[0])
        #weights = torch.randn(2, self.hidden_size[0])
        self.room_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        
        # lstm encoder
        self.encoder_lstm = torch.nn.LSTM(self.x_size, self.hidden_size[0], batch_first=True, num_layers=hyperparameters["num_layers"])
        # lstm decoder
        self.proj_size = 1
        self.decoder_lstm = torch.nn.LSTM(self.y_features_size, self.hidden_size[0], proj_size=self.proj_size, batch_first=True, num_layers=hyperparameters["num_layers"])
        # predictor
        self.linear = torch.nn.Linear(self.hidden_size[0], hyperparameters["y_size"])
        
        if self.occcount:
            self.last_activation = nn.ReLU()    
        else:
            self.last_activation = nn.Sigmoid()
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # embedding for hidden states with 0 init
        weights = torch.randn(2, self.hidden_size[0])
        #weights = torch.zero(2, self.hidden_size[0])
        #weights = torch.randn(2, self.hidden_size[0])
        self.room_embedding = nn.Embedding.from_pretrained(weights, freeze=False)       
             
    def forward(self, x, y_features, room_id=None):
        
        room_enc = self.room_embedding(room_id)[None, :, :]
        room_enc = room_enc.repeat(self.hyperparameters["num_layers"], 1, 1)
        
        _, (_, c_n) = self.encoder_lstm(x, (torch.zeros(self.hyperparameters["num_layers"], self.batch_size, self.hidden_size[0]).to(self.device), room_enc))
        
        out, _ = self.decoder_lstm(y_features, (torch.zeros(self.hyperparameters["num_layers"], self.batch_size, self.proj_size).to(self.device), c_n))
                
        return out.squeeze(-1)






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
class EncDecOccLSTM1(torch.nn.Module):
    
    def __init__(self, hyperparameters, **kwargs):
        
        super().__init__()
        
        self.hyperparameters = hyperparameters

        self.x_size = hyperparameters["x_size"]
        self.y_features_size = hyperparameters["y_features_size"]
        #self.input_size = self.x_size + self.y_features_size 
    
        self.hidden_size = hyperparameters["hidden_size"]
        
        self.occcount = hyperparameters["occcount"]
        self.batch_size = hyperparameters["batch_size"]
        
        self.output_size = hyperparameters["y_size"] * hyperparameters["y_horizon"]
        
        # lstm encoder
        self.encoder_lstm = torch.nn.LSTM(self.x_size, self.hidden_size[0], batch_first=True, num_layers=hyperparameters["num_layers"])
        # lstm decoder
        self.decoder_lstm = torch.nn.LSTM(self.y_features_size, self.hidden_size[0], proj_size=1, batch_first=True, num_layers=hyperparameters["num_layers"])
        
        #print(self.x_size, self.hidden_size[0])
        #print(self.decoder_lstm.weight_ih_l0.shape)
        #print(self.decoder_lstm.weight_ih_l1.shape)
        ## Build custom lstm!
        #raise ValueError("Stop")
        #print(self.decoder_lstm)
        #raise ValueError("Stop")
        if self.occcount:
            self.last_activation = nn.ReLU()    
        else:
            self.last_activation = nn.Sigmoid()
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
    def forward(self, x, y_features):
        
        _, (_, c_n) = self.encoder_lstm(x)
        
        out, (_, _) = self.decoder_lstm(
            y_features, (
                torch.zeros(self.hyperparameters["num_layers"], self.batch_size, 1).to(self.device), 
                c_n
                )
            )
        
        pred = self.last_activation(out)
        return pred.squeeze()
    
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
