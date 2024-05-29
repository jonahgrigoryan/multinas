import pandas as pd
import numpy as np
import pyro
import optuna
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam  
from pyro.nn import PyroParam
import pyro.distributions as dist
import pyro.nn as pynn

data = pd.read_parquet('finalscaled.parquet')                                                                    
features = data.drop('target', axis=1)
target = data['target']

# Define evaluation metrics
def correlation_coefficient(predictions, targets):
    """Calculate the correlation coefficient between predictions and targets."""
    return np.corrcoef(predictions, targets)[0, 1]

def feature_exposure(predictions, features):
    """Calculate the feature exposure of predictions to features."""
    return np.mean([np.corrcoef(predictions, features[:, i])[0, 1] for i in range(features.shape[1])])

def mmc(predictions, targets):
    """Calculate the mean-variance optimization metric."""
    return np.mean(predictions * targets) / np.std(predictions * targets)

def sharpe_ratio(predictions, targets):
    """Calculate the Sharpe ratio of predictions."""
    return np.mean(predictions * targets) / np.std(predictions * targets)

# Example function to evaluate model
def evaluate_model(model, data_loader):
    predictions = []
    targets = []
    for features, target in data_loader:
        features = features.to(torch.float32)
        target = target.to(torch.float32)
        pred = model(features).detach().numpy()
        predictions.append(pred)
        targets.append(target.numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    results = {
        'correlation_coefficient': correlation_coefficient(predictions, targets),
        'feature_exposure': feature_exposure(predictions, data_loader.dataset.tensors[0].numpy()),
        'mmc': mmc(predictions, targets),
        'sharpe_ratio': sharpe_ratio(predictions, targets)
    }
    
    return results

# Define the search space for hyperparameters using Optuna
def define_search_space(trial):
    search_space = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'epochs': trial.suggest_int('epochs', 50, 100),
        'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'swish']),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'adadelta', 'nadam', 'ftrl']),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
        'weight_initialization': trial.suggest_categorical('weight_initialization', ['xavier', 'he', 'orthogonal']),
        'architecture': trial.suggest_categorical('architecture', ['mlp', 'brnn', 'bvae', 'mc_dropout', 'vbnn']),
        'mlp_layers': trial.suggest_int('mlp_layers', 2, 5),
        'rnn_layers': trial.suggest_int('rnn_layers', 2, 4),
        'vae_latent_dim': trial.suggest_int('vae_latent_dim', 10, 50),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'hidden_dim': trial.suggest_int('hidden_dim', 50, 200)
    }
    return search_space


# Define the MLP model using Pyro
class MLP(pynn.PyroModule):
    def __init__(self, input_dim, hidden_dims, output_dim, activation, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(pynn.PyroModule[nn.Linear](input_dim, hidden_dims[0]))
        self.layers.append(self.get_activation(activation))
        self.layers.append(nn.Dropout(dropout_rate))
        
        for i in range(1, len(hidden_dims)):
            self.layers.append(pynn.PyroModule[nn.Linear](hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(self.get_activation(activation))
            self.layers.append(nn.Dropout(dropout_rate))
        
        self.output_layer = pynn.PyroModule[nn.Linear](hidden_dims[-1], output_dim)

    def get_activation(self, activation):
        return {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'swish': nn.SiLU()
        }[activation]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f"Layer {i} output shape: {x.shape}")  # Debugging: print shape after each layer
        x = self.output_layer(x)
        print(f"Output layer shape: {x.shape}")  # Debugging: print shape of output layer
        return x

# Define the model function for MLP
def model_mlp(trial, input_dim, output_dim):
    # Sample hyperparameters
    hidden_layers = trial.suggest_int('mlp_layers', 2, 5)
    hidden_dims = [trial.suggest_int(f'hidden_dim_{i}', 50, 200) for i in range(hidden_layers)]
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'swish'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)

    # Debugging: Print the sampled hyperparameters
    print(f"Sampled hyperparameters: hidden_layers={hidden_layers}, hidden_dims={hidden_dims}, activation={activation}, dropout_rate={dropout_rate}")

    # Instantiate MLP model
    mlp_model = MLP(input_dim, hidden_dims, output_dim, activation, dropout_rate)

    # Define Pyro model
    def model(features, targets=None):
        pyro.module("mlp_model", mlp_model)

        # Sample weights and biases for each layer
        for i, layer in enumerate(mlp_model.layers):
            if isinstance(layer, pynn.PyroModule[nn.Linear]):
                weight_prior = dist.Normal(0, 1).expand(layer.weight.shape).to_event(2)
                bias_prior = dist.Normal(0, 1).expand(layer.bias.shape).to_event(1)
                layer.weight = pyro.nn.PyroParam(pyro.sample(f"w_{i}", weight_prior))
                layer.bias = pyro.nn.PyroParam(pyro.sample(f"b_{i}", bias_prior))

        # Sample weights and biases for output layer
        weight_prior = dist.Normal(0, 1).expand(mlp_model.output_layer.weight.shape).to_event(2)
        bias_prior = dist.Normal(0, 1).expand(mlp_model.output_layer.bias.shape).to_event(1)
        mlp_model.output_layer.weight = pyro.nn.PyroParam(pyro.sample("w_output", weight_prior))
        mlp_model.output_layer.bias = pyro.nn.PyroParam(pyro.sample("b_output", bias_prior))

        # Define the likelihood
        with pyro.plate("data", features.shape[0]):
            outputs = mlp_model(features).squeeze(-1)  # Ensure the outputs have the same shape as targets
            pyro.sample("obs", dist.Normal(outputs, 1).to_event(1), obs=targets)

    return model

# Example usage of model_mlp in an objective function
# (actual implementation of objective function should follow)
# input_dim and output_dim need to be defined based on dataset
input_dim = 214  # excluding target column
output_dim = 1  # assuming single target variable

class BRNN(pyro.nn.PyroModule):
    def __init__(self, input_dim, hidden_dim, rnn_layers, rnn_type, dropout_rate):
        super().__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        if rnn_type == 'lstm':
            self.rnn = pyro.nn.PyroModule[nn.LSTM](input_dim, hidden_dim, rnn_layers, batch_first=True, dropout=dropout_rate)
        else:
            self.rnn = pyro.nn.PyroModule[nn.GRU](input_dim, hidden_dim, rnn_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = pyro.nn.PyroModule[nn.Linear](hidden_dim, 1)
        
        # Print the configuration for debugging
        print(f"BRNN Configuration:\nInput Dim: {input_dim}\nHidden Dim: {hidden_dim}\nRNN Layers: {rnn_layers}\nRNN Type: {rnn_type}\nDropout Rate: {dropout_rate}")

    def forward(self, x, sample_site=None):
        print(f"Input shape: {x.shape}")
        if self.rnn_type == 'lstm' or self.rnn_type == 'gru':
            print(f"RNN Input shape before forward pass: {x.shape}")
            out, _ = self.rnn(x)
            print(f"RNN Output shape after forward pass: {out.shape}")
        else:
            raise ValueError("Invalid RNN type: must be 'lstm' or 'gru'")
        out = self.dropout(out[:, -1, :])  # Use the output of the last time step
        print(f"Shape after dropout: {out.shape}")
        out = self.output_layer(out)
        print(f"Output shape: {out.shape}")
        return out

# Define the model function for BRNN
def model_brnn(trial, input_dim, output_dim):
    # Sample hyperparameters
    rnn_layers = trial.suggest_int('rnn_layers', 2, 4)
    hidden_dim = trial.suggest_int('hidden_dim', 50, 200)
    rnn_type = trial.suggest_categorical('rnn_type', ['lstm', 'gru'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)

    # Instantiate BRNN model
    brnn_model = BRNN(input_dim, hidden_dim, rnn_layers, rnn_type, dropout_rate)

    # Define Pyro model
    def model(features, targets=None):
        pyro.module("brnn_model", brnn_model)

        # Sample weights and biases for output layer
        weight_prior = dist.Normal(0, 1).expand(brnn_model.output_layer.weight.shape).to_event(2)
        bias_prior = dist.Normal(0, 1).expand(brnn_model.output_layer.bias.shape).to_event(1)
        brnn_model.output_layer.weight = pyro.nn.PyroParam(pyro.sample("w_output", weight_prior))
        brnn_model.output_layer.bias = pyro.nn.PyroParam(pyro.sample("b_output", bias_prior))

        # Define the likelihood
        with pyro.plate("data", features.shape[0]):
            outputs = brnn_model(features)
            pyro.sample("obs", dist.Normal(outputs, 1), obs=targets)

    return brnn_model, model

# Example usage of model_brnn in an objective function
# (actual implementation of objective function should follow)
# input_dim and output_dim need to be defined based on dataset
input_dim = 214  # Define input_dim correctly based on your data
output_dim = 1  # assuming single target variable

# Define the Encoder part of the BVAE
class Encoder(pynn.PyroModule):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = pynn.PyroModule[nn.Linear](input_dim, hidden_dim)
        self.fc2_mean = pynn.PyroModule[nn.Linear](hidden_dim, latent_dim)
        self.fc2_logvar = pynn.PyroModule[nn.Linear](hidden_dim, latent_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        h = self.activation(self.fc1(x))
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        return mean, logvar

# Define the Decoder part of the BVAE
class Decoder(pynn.PyroModule):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = pynn.PyroModule[nn.Linear](latent_dim, hidden_dim)
        self.fc2 = pynn.PyroModule[nn.Linear](hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, z):
        h = self.activation(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon

# Define the BVAE model using Pyro
class BVAE(pynn.PyroModule):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.latent_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.latent_dim)))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            x_recon = self.decoder(z)
            pyro.sample("obs", dist.Bernoulli(x_recon).to_event(1), obs=x)

    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_logvar = self.encoder(x)
            z_scale = torch.exp(z_logvar * 0.5)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def forward(self, x, *args, **kwargs):
        z_loc, z_logvar = self.encoder(x)
        z_scale = torch.exp(z_logvar * 0.5)
        z = dist.Normal(z_loc, z_scale).rsample()
        return self.decoder(z)


# Define the model function for BVAE
def model_bvae(trial, input_dim, output_dim):
    # Sample hyperparameters
    hidden_dim = trial.suggest_int('hidden_dim', 50, 200)
    latent_dim = trial.suggest_int('vae_latent_dim', 10, 50)

    # Instantiate BVAE model
    bvae_model = BVAE(input_dim, hidden_dim, latent_dim)

    return bvae_model

# Example usage of model_bvae in an objective function
# (actual implementation of objective function should follow)
# input_dim and output_dim need to be defined based on dataset
input_dim = data.shape[1] - 1  # excluding target column
output_dim = 1  # assuming single target variable
                                                                                        
# Define the MCDropout model using Pyro                                                                          
class MCDropout(pynn.PyroModule):                                                                                
  def __init__(self, input_dim, hidden_dims, output_dim, activation, dropout_rate):                            
      super().__init__()                                                                                       
      self.layers = nn.ModuleList()                                                                            
      self.layers.append(pynn.PyroModule[nn.Linear](input_dim, hidden_dims[0]))                                
      self.layers.append(nn.ReLU() if activation == 'relu' else                                                
                         nn.LeakyReLU() if activation == 'leaky_relu' else                                     
                         nn.ELU() if activation == 'elu' else                                                  
                         nn.SiLU())                                                                            
      self.layers.append(nn.Dropout(dropout_rate))                                                             
                                                                                                               
      for i in range(1, len(hidden_dims)):                                                                     
          self.layers.append(pynn.PyroModule[nn.Linear](hidden_dims[i-1], hidden_dims[i]))                     
          self.layers.append(nn.ReLU() if activation == 'relu' else                                            
                             nn.LeakyReLU() if activation == 'leaky_relu' else                                 
                             nn.ELU() if activation == 'elu' else                                              
                             nn.SiLU())                                                                        
          self.layers.append(nn.Dropout(dropout_rate))                                                         
                                                                                                               
      self.output_layer = pynn.PyroModule[nn.Linear](hidden_dims[-1], output_dim)                              
                                                                                                               
  def forward(self, x):                                                                                        
      for layer in self.layers:                                                                                
          x = layer(x)                                                                                         
      x = self.output_layer(x)                                                                                 
      return x                                                                                                 
                                                                                                               
# Define the model function for MCDropout                                                                        
def model_mc_dropout(trial, input_dim, output_dim):                                                              
  # Sample hyperparameters                                                                                     
  hidden_layers = trial.suggest_int('mlp_layers', 2, 5)                                                        
  hidden_dims = [trial.suggest_int(f'hidden_dim_{i}', 50, 200) for i in range(hidden_layers)]                  
  activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'swish'])                 
  dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)                                                 
                                                                                                               
  # Instantiate MCDropout model                                                                                
  mc_dropout_model = MCDropout(input_dim, hidden_dims, output_dim, activation, dropout_rate)                   
                                                                                                               
  # Define Pyro model                                                                                          
  def model(features, targets=None):                                                                           
      # Sample weights and biases for each layer                                                               
      for i, layer in enumerate(mc_dropout_model.layers):                                                                    
          if isinstance(layer, pynn.PyroModule[nn.Linear]):                                                    
              weight_prior = dist.Normal(0, 1).expand(layer.weight.shape).to_event(2)                          
              bias_prior = dist.Normal(0, 1).expand(layer.bias.shape).to_event(1)                              
              layer.weight = pyro.nn.PyroParam(pyro.sample(f"w_{i}", weight_prior))                                           
              layer.bias = pyro.nn.PyroParam(pyro.sample(f"b_{i}", bias_prior))                                               
                                                                                                               
      # Sample weights and biases for output layer                                                             
      weight_prior = dist.Normal(0, 1).expand(mc_dropout_model.output_layer.weight.shape).to_event(2)          
      bias_prior = dist.Normal(0, 1).expand(mc_dropout_model.output_layer.bias.shape).to_event(1)              
      mc_dropout_model.output_layer.weight = pyro.nn.PyroParam(pyro.sample("w_output", weight_prior))                             
      mc_dropout_model.output_layer.bias = pyro.nn.PyroParam(pyro.sample("b_output", bias_prior))                                 
                                                                                                               
      # Define the likelihood                                                                                  
      with pyro.plate("data", features.shape[0]):                                                              
          outputs = mc_dropout_model(features)                                                                 
          pyro.sample("obs", dist.Normal(outputs, 1), obs=targets)                                             
                                                                                                               
  return model                                                                                                 

# Define the VBNNs model using Pyro
class VBNNs(pynn.PyroModule):
    def __init__(self, input_dim, hidden_dims, output_dim, activation, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(pynn.PyroModule[nn.Linear](input_dim, hidden_dims[0]))
        self.layers.append(self.get_activation(activation))
        self.layers.append(nn.Dropout(dropout_rate))

        for i in range(1, len(hidden_dims)):
            self.layers.append(pynn.PyroModule[nn.Linear](hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(self.get_activation(activation))
            self.layers.append(nn.Dropout(dropout_rate))

        self.output_layer = pynn.PyroModule[nn.Linear](hidden_dims[-1], output_dim)

    def get_activation(self, activation):
        return {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'swish': nn.SiLU()
        }[activation]

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Reshape the input tensor to 2D
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

# Define the model function for VBNNs
def model_vbnn(trial, input_dim, output_dim):
    # Sample hyperparameters
    hidden_layers = trial.suggest_int('mlp_layers', 2, 5)
    hidden_dims = [trial.suggest_int(f'hidden_dim_{i}', 50, 200) for i in range(hidden_layers)]
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'swish'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)

    # Instantiate VBNNs model
    vbnn_model = VBNNs(input_dim, hidden_dims, output_dim, activation, dropout_rate)

    # Define Pyro model
    def model(features, targets=None):
        pyro.module("vbnn_model", vbnn_model)

        with pyro.plate("data", features.shape[0]):
            # Sample weights and biases for each layer
            for i, layer in enumerate(vbnn_model.layers):
                if isinstance(layer, pynn.PyroModule[nn.Linear]):
                    weight_prior = dist.Normal(0, 1).expand(layer.weight.shape).to_event(2)
                    bias_prior = dist.Normal(0, 1).expand(layer.bias.shape).to_event(1)
                    layer.weight = pyro.nn.PyroParam(pyro.sample(f"w_{i}", weight_prior))
                    layer.bias = pyro.nn.PyroParam(pyro.sample(f"b_{i}", bias_prior))

            # Sample weights and biases for output layer
            weight_prior = dist.Normal(0, 1).expand(vbnn_model.output_layer.weight.shape).to_event(2)
            bias_prior = dist.Normal(0, 1).expand(vbnn_model.output_layer.bias.shape).to_event(1)
            vbnn_model.output_layer.weight = pyro.nn.PyroParam(pyro.sample("w_output", weight_prior))
            vbnn_model.output_layer.bias = pyro.nn.PyroParam(pyro.sample("b_output", bias_prior))

            # Define the likelihood
            print("Input data shape:", features.shape)
            outputs = vbnn_model(features)
            pyro.sample("obs", dist.Normal(outputs, 1), obs=targets)

    return model

# Example usage of model_vbnn in an objective function
# (actual implementation of objective function should follow)
# input_dim and output_dim need to be defined based on dataset
input_dim = 214  # Example fixed input dimension based on dummy data
output_dim = 1  # Assuming single target variable

# Define the objective function that evaluates the performance of each architecture
def objective(trial):
    # Sample hyperparameters and architecture type
    params = define_search_space(trial)
    architecture = params['architecture']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    epochs = params['epochs']
    activation = params['activation']
    dropout_rate = params['dropout_rate']

    # Define input and output dimensions
    input_dim = data.shape[1] - 1  # excluding target column
    output_dim = 1  # assuming single target variable

    # Select and instantiate the model based on the architecture type
    # Select and instantiate the model based on the architecture type
    if architecture == 'mlp':
        model_fn = model_mlp(trial, input_dim, output_dim)
    elif architecture == 'brnn':
        model_fn, guide_fn = model_brnn(trial, input_dim, output_dim)
    elif architecture == 'bvae':
        model_fn = model_bvae(trial, input_dim, output_dim)
    elif architecture == 'mc_dropout':
        model_fn = model_mc_dropout(trial, input_dim, output_dim)
    elif architecture == 'vbnn':
        model_fn = model_vbnn(trial, input_dim, output_dim)
    
    # Define Pyro guide (e.g., AutoDiagonalNormal)
    if architecture == 'brnn':
        guide = guide_fn
    else:
        guide = pyro.infer.autoguide.AutoDiagonalNormal(model_fn)


    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data.drop(columns=['target']), data['target'], 
                                                      test_size=0.2, random_state=42)

    # Create DataLoader for PyTorch
    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), 
                                  torch.tensor(y_train.values, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32), 
                                torch.tensor(y_val.values, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define Pyro guide (e.g., AutoDiagonalNormal)
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model_fn)

    # Define optimizer and SVI
    pyro_optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(model_fn, guide, pyro_optimizer, loss=Trace_ELBO())

    # Training loop
    for epoch in range(epochs):
        for features, target in train_loader:
            svi.step(features, target)

    # Evaluation
    results = evaluate_model(model_fn, val_loader)

    # Return the main evaluation metric for optimization
    return results['correlation_coefficient']

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print best trial
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
