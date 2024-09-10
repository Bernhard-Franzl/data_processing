import torch
import torch.nn as nn

class SimpleOccDenseNet(nn.Module):
    
    def __init__(self, hyperparameters):
        super().__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.hyperparameters = hyperparameters

        self.x_size = hyperparameters["x_size"] * hyperparameters["x_horizon"]
        self.y_horizon = hyperparameters["y_horizon"]
        self.y_features_size = hyperparameters["y_features_size"] * hyperparameters["y_horizon"]
    
        self.output_size = hyperparameters["y_size"] * hyperparameters["y_horizon"]
    
        self.hidden_size = hyperparameters["hidden_size"]
        
        self.occcount = hyperparameters["occcount"]
        self.batch_size = hyperparameters["batch_size"]
        
        self.room_encoding = torch.eye(2).to(self.device)
        self.room_enc_size = self.room_encoding.shape[1]
        
        self.input_size = self.x_size + self.y_features_size + self.room_enc_size
        
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
                self.model.add_module("acti_out", nn.Sigmoid())
                #print("Sigmoid added")
   
    def forward(self, x, y_features, room_id=None):
        
        room_enc = self.room_encoding[room_id]
        room_enc = room_enc.view(self.batch_size, -1)
        
        x = x.view(self.batch_size, -1)
        y_features = y_features.view(self.batch_size, -1)
        input = torch.cat((x, y_features, room_enc), 1)

        out = self.model(input)
        return out
    
    def forecast_iter(self, x, y_features, len_y, room_id=None):
        
        room_enc = self.room_encoding[room_id].to("cpu")

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
            
            predicitons.append(out)

            if self.hyperparameters["include_x_features"]:
                x_new = torch.cat((out[:, None], y_feat_i), dim=1)
            else:
                x_new = out[:, None]
                
            x = torch.cat((x, x_new), dim=0)[self.y_horizon:]

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
           

        # lstm layer
        self.lstm = torch.nn.LSTM(self.x_size, self.hidden_size[0], batch_first=True, num_layers=hyperparameters["num_layers"])
        # fc at end
        self.linear_final = torch.nn.Linear(self.hidden_size[0], 1)
        
        #self.linear2 = torch.nn.Linear(self.hidden_size[0], self.output_size)
        self.linear_in = torch.nn.Linear(self.y_features_size, self.hidden_size[0])
    
        if self.occcount:
            self.last_activation = nn.ReLU()    
        else:
            self.last_activation = nn.Sigmoid()
        
    def forward(self, x, y_features,room_id=None):
        
        room_enc = self.room_embedding(room_id)[None, :, :]
        room_enc = room_enc.repeat(self.hyperparameters["num_layers"], 1, 1)
        out, (h_n, c_n) = self.lstm(x, (torch.zeros(self.hyperparameters["num_layers"], self.batch_size, self.hidden_size[0]).to(self.device), room_enc))       
        
        y_t = self.last_activation(self.linear_final(out[:, -1, :]))  
        y_t = y_t[:, None, :]
        
        pred_list = []
        for i in range(0, self.hyperparameters["y_horizon"]):
            
            y_feat_t = y_features[:, i, :]
            y_in = torch.cat((y_t, y_feat_t[:, None, :]), 2)
            
            h_t1, (h_n, c_n) = self.lstm(y_in, (h_n, c_n))
            y_t = self.last_activation(self.linear_final(h_t1))

            pred_list.append(y_t)
        
        pred_list = torch.cat(pred_list, dim=1).squeeze(-1)
        
        return pred_list
    
    def forecast_iter(self, x, y_features, len_y, room_id):
        
        x = x[None, :]
        y_features = y_features[None, :]
        
        room_enc = self.room_embedding(room_id)
        room_enc = room_enc.repeat(self.hyperparameters["num_layers"], 1, 1).to("cpu")

        
        out, (h_n, c_n) = self.lstm(x, (torch.zeros(self.hyperparameters["num_layers"], 1, self.hidden_size[0]).to("cpu"), room_enc))       
    
        y_t = self.last_activation(self.linear_final(out[:, -1, :]))[:, None, :]
        pred_list = []
        for i in range(0, len_y):
            
            y_feat_i = y_features[:, i, :][None, :]
            y_t_in = torch.cat((y_t, y_feat_i), 2)
            
            out, (h_n, c_n) = self.lstm(y_t_in, (h_n, c_n))

            y_t = self.last_activation(self.linear_final(out))

            pred_list.append(y_t)

        return torch.cat(pred_list).squeeze()

    #def forward(self, x, y_features, room_id=None):
        
    #    h_t = torch.zeros(self.batch_size, self.hidden_size[0]).to(self.device)
    #    c_t = self.room_embedding(room_id)
        
    #    for i in range(self.hyperparameters["x_horizon"]):
    #        x_t = x[:, i, :]
    #        h_t, c_t = self.lstm_cell(x_t, (h_t, c_t))
         
    #    y_t = self.last_activation(self.linear_final(h_t)) 
        
    #    y_t_list = []
    #    for i in range(0, self.hyperparameters["y_horizon"]):
            
    #        y_feat_t = y_features[:, i, :]
    #        y_in = torch.cat((y_t, y_feat_t), 1)
            
    #        h_t, c_t = self.lstm_cell(y_in, (h_t, c_t))
    #        y_t = self.last_activation(self.linear_final(h_t))

    #        y_t_list.append(y_t)
            
    #    pred = torch.cat(y_t_list, dim=1)
    #    return pred

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
