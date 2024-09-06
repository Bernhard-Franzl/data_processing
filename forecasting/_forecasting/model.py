import torch
import torch.nn as nn

class SimpleOccDenseNet(nn.Module):
    
    def __init__(self, hyperparameters):
        super().__init__()
        
        self.hyperparameters = hyperparameters

        self.x_size = hyperparameters["x_size"] * hyperparameters["x_horizon"]
        self.y_features_size = hyperparameters["y_features_size"] * hyperparameters["y_horizon"]
        self.input_size = self.x_size + self.y_features_size 
    
        self.output_size = hyperparameters["y_size"] * hyperparameters["y_horizon"]
    
        self.hidden_size = hyperparameters["hidden_size"]
        
        self.occcount = hyperparameters["occcount"]
        self.batch_size = hyperparameters["batch_size"]
        
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
                self.model.add_module("relu", nn.ReLU())
                self.model.add_module("output_layer", nn.Linear(self.hidden_size[0], self.output_size))
            
        if self.occcount:
            self.model.add_module("sigmoid", nn.ReLU())
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
        
        self.hyperparameters = hyperparameters

        self.x_size = hyperparameters["x_size"]
        self.y_features_size = hyperparameters["y_features_size"]
    
        self.output_size = hyperparameters["y_size"] * hyperparameters["y_horizon"]
    
        self.hidden_size = hyperparameters["hidden_size"]
        
        self.occcount = hyperparameters["occcount"]
        self.batch_size = hyperparameters["batch_size"]
        

        # lstm layer
        self.lstm = torch.nn.LSTM(self.x_size, self.hidden_size[0], batch_first=True, num_layers=hyperparameters["num_layers"])
        # fc at end
        self.linear_final = torch.nn.Linear(self.hidden_size[0] + self.y_features_size, self.output_size)
        
        #self.linear2 = torch.nn.Linear(self.hidden_size[0], self.output_size)
    
        if self.occcount:
            self.last_activation = nn.ReLU()    
        else:
            self.last_activation = nn.Sigmoid()
            
    def forward(self, x, y_features):
        
        out, _ = self.lstm(x)
        
        out = out[:, -1, :]
        
        y_features = y_features.view(-1, self.y_features_size*self.hyperparameters["y_horizon"])

        flattened = torch.cat((out, y_features), 1)
        pred = self.last_activation(self.linear_final(flattened))

        return pred

class EncDecOccLSTM(torch.nn.Module):
    
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
        self.encoder_lstm = torch.nn.LSTM(self.x_size, self.hidden_size[0], batch_first=True)
        # lstm decoder
        self.decoder_lstm = torch.nn.LSTM(self.y_features_size, self.hidden_size[0], batch_first=True)
        # predictor
        self.linear = torch.nn.Linear(self.hidden_size[0], hyperparameters["y_size"])
        
        if self.occcount:
            self.last_activation = nn.ReLU()    
        else:
            self.last_activation = nn.Sigmoid()
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
    def forward(self, x, y_features):
        
        _, (_, c_n) = self.encoder_lstm(x)
        
        out, _ = self.decoder_lstm(y_features, (torch.zeros(1, self.batch_size, self.hidden_size[0]).to(self.device), c_n))
        
        pred_list = [ self.last_activation(self.linear(out[:, i, :])) for i in range(self.hyperparameters["y_horizon"])]
                
        return torch.stack(pred_list, 1).squeeze()

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
