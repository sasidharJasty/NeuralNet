import pickle
import numpy as np
from dense import Dense
from dense_adam import DenseAdam
from activations import Tanh

def save_model(network, filepath, x_mean=None, x_std=None):
    """Save the trained network and normalization parameters"""
    model_weights = []
    layer_types = []
    
    for layer in network:
        if isinstance(layer, Dense):
            model_weights.append({
                'weights': layer.weights,
                'bias': layer.bias,
                'input_size': layer.weights.shape[1],
                'output_size': layer.weights.shape[0],
                'layer_type': 'Dense'
            })
            layer_types.append('Dense')
        elif isinstance(layer, DenseAdam):
            model_weights.append({
                'weights': layer.weights,
                'bias': layer.bias,
                'input_size': layer.weights.shape[1],
                'output_size': layer.weights.shape[0],
                'alpha': layer.alpha,
                'beta1': layer.beta1,
                'beta2': layer.beta2,
                'epsilon': layer.epsilon,
                'm_w': layer.m_w,
                'v_w': layer.v_w,
                'm_b': layer.m_b,
                'v_b': layer.v_b,
                't': layer.t,
                'layer_type': 'DenseAdam'
            })
            layer_types.append('DenseAdam')
        elif isinstance(layer, Tanh):
            layer_types.append('Tanh')
    
    model_data = {
        'model_weights': model_weights,
        'layer_types': layer_types,
        'x_mean': x_mean,
        'x_std': x_std
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load a trained network and normalization parameters"""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    network = []
    weight_idx = 0
    
    for layer_type in model_data['layer_types']:
        if layer_type == 'Dense':
            weights_data = model_data['model_weights'][weight_idx]
            layer = Dense(weights_data['input_size'], weights_data['output_size'])
            layer.weights = weights_data['weights']
            layer.bias = weights_data['bias']
            network.append(layer)
            weight_idx += 1
        elif layer_type == 'DenseAdam':
            weights_data = model_data['model_weights'][weight_idx]
            layer = DenseAdam(
                weights_data['input_size'], 
                weights_data['output_size'],
                weights_data['alpha'],
                weights_data['beta1'],
                weights_data['beta2'],
                weights_data['epsilon']
            )
            layer.weights = weights_data['weights']
            layer.bias = weights_data['bias']
            layer.m_w = weights_data['m_w']
            layer.v_w = weights_data['v_w']
            layer.m_b = weights_data['m_b']
            layer.v_b = weights_data['v_b']
            layer.t = weights_data['t']
            network.append(layer)
            weight_idx += 1
        elif layer_type == 'Tanh':
            network.append(Tanh())
    
    print(f"Model loaded from {filepath}")
    return network, model_data['x_mean'], model_data['x_std']

def force_retrain():
    import os
    model_file = "iris_neural_network_adam.pkl"
    if os.path.exists(model_file):
        os.remove(model_file)
        print(f"Deleted {model_file}. Run the script again to retrain.")