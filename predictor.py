#@title loading utils
import argparse
import torch
from omegaconf import OmegaConf
from AE.util import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler
# import sys
import os
import numpy as np 
from utils_omar import load_arrays

#%%  Parser Functions 
def parse_arguments():
    parser = argparse.ArgumentParser(description="A simple script to demonstrate argument parsing.")
    parser.add_argument('-l', '--logdir', type=str, help='Directory Contains the logs and check points', required=True)  # Where to load the model and save the predictions
    parser.add_argument('-nf', '--num_fr', type=int, help='Number of frames to do the inference', required=True)  # Number of frames to predict 
    parser.add_argument('-test_path', '--test_path', type=str, help='Path contains test data to predict', required=True)  #  From where to load the test data 

    

    return parser.parse_args()

args = parse_arguments()
path_to_logs = args.logdir 
path_test_data = args.test_path 

num_files  = args.num_fr
   




#%%
def find_project_yaml(directory):
    try:
        for file in os.listdir(directory):
            if file.endswith('project.yaml'):
                return file
        return None
    except FileNotFoundError:
        return "Directory not found."
    except Exception as e:
        return f"An error occurred: {e}"

#%%
def get_folder_names(directory):
    try:
        folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        return folders
    except FileNotFoundError:
        return "Directory not found."
    except Exception as e:
        return f"An error occurred: {e}"

#%%
def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model(path):
    # path  = 
    config_files_path =  path + '/configs/'
    config_file_path =config_files_path+ find_project_yaml(config_files_path)
    config = OmegaConf.load(config_file_path)  
    model = load_model_from_config(config, path + "/checkpoints/last.ckpt")
    return model
#%%  Chose the cases to plot 
#path_to_logs = args.logdir '/projects/ai4wind/osallam/inflows/DiT/logs_Not_Normalized/' 

cases = get_folder_names(path_to_logs)
#%% Trials to plot 

case_counter  = 0
for case_ in cases:
    
    
    path = path_to_logs + case_ + '/'
    model = get_model(path)
    
    #%% Make directory for inference 

    Inference_path = path+'Inference/'
    if not os.path.exists(Inference_path):
        os.makedirs(Inference_path)
    
    #%%   Load Test data 
    
    if case_counter ==0:
        test_data_numpy = load_arrays(path_test_data, num_files)  # Load the data to CPU
        test_data_numpy = np.array(test_data_numpy, dtype=np.float32)   # Convert to single percision
        
        
        
    pred_data = np.zeros_like(test_data_numpy)
    
    test_data = torch.from_numpy(test_data_numpy)    # Make a torch tensor from the numpy
    test_data = test_data.to('cuda')  # Move input tensor to GPU
    
    # pred_data = torch.zeros_like(test_data)
    
    
    #%%
    model.eval()
    with torch.no_grad():
        pred_data= model(test_data)[0].cpu().numpy()
    np.save(Inference_path + 'pred_Infer.npy', pred_data)
    np.save(Inference_path + 'test_Infer.npy', test_data_numpy)
    print('---------------------------------\n')
    print(f'Inference done for case {case_counter} in folder'+ Inference_path )
    print('---------------------------------\n')

    
    # Return the test data back to the cpu
    test_data = test_data.cpu().numpy()
    
        
    
    
    case_counter +=1
    
    






