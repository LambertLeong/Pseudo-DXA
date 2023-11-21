"""
File Name: predict.py
Author: Lambert T Leong
Description: This code loads 3D meshes, fitted/standardized using meshcapade, as a *.ply file, loads the Pseudo-DXA model, runs the mesh through the model, predicts a DXA scan, and outputs the raw DXA data.
"""

import os, sys,configparser,importlib, ast,argparse, cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1: WARNING, 2: ERROR, 3: FATAL)
from argparse import RawTextHelpFormatter
import tensorflow as tf
from functools import reduce
import numpy as np

sys.path.append('utils/')
import DXA_utils as dxa
import mesh_utils as mesh

sys.path.append('modeling/')
import mesh_encoder
import dxa_generator
import pseudo_DXA

def parse_arguments():
    parser = argparse.ArgumentParser(epilog="""
    list index (--config, -c)
            - path to config file 
     """,
    formatter_class=RawTextHelpFormatter)
    parser.add_argument('--config', '-c', action='store',
    required=True, dest='config', metavar='<config file>',
    help='"path to config file"')
    return parser.parse_args()

def main():
    argv = parse_arguments()
    configfile = str(argv.config)
    config = configparser.ConfigParser()
    config.read(configfile)	
    desired_gpu_index = int(config['setup']['gpu'])
    
   # Check for available GPU devices
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        if desired_gpu_index < len(gpu_devices):
            # Use the desired GPU by setting it as the visible device
            tf.config.experimental.set_visible_devices(gpu_devices[desired_gpu_index], 'GPU')
            print(f"Using GPU {desired_gpu_index}: {gpu_devices[desired_gpu_index].name}")
        else:
            # Use the first available GPU if the desired index is out of range
            tf.config.experimental.set_visible_devices(gpu_devices[0], 'GPU')
            print(f"Desired GPU index {desired_gpu_index} is out of range. Using GPU 0: {gpu_devices[0].name}")
    else:
        # No GPU available, use CPU
        print("No CUDA-enabled GPU found. Using CPU.")
        
    os.environ["CUDA_VISIBLE_DEVICES"]=str(desired_gpu_index)
    input_size = int(config['model']['input_size'])
    upscale_factor = int(config['model']['upscale'])
    reshape_dims= [int(x) for x in config['model']['reshape_dims'].split(',')]
    psdxa_weights=config['model']['lleong_psdxa_weights']
    latent_dim = reduce((lambda x, y: x * y), reshape_dims)
    outpath = config['predict']['out_path']

    print('Loading Model and Weights')
    encoder = mesh_encoder.MeshEncoder(latent_dim,num_pts=input_size).model
    generator = dxa_generator.DXAGenerator(reshape_dim=reshape_dims, upscale=upscale_factor).model
    psdxa = pseudo_DXA.PseudoDXA(encoder,generator)
    psdxa.build_model()
    psdxa.model.load_weights(psdxa_weights)

    print('Loading and Processing Data')
    npy=mesh.ply_to_npy(config['predict']['mesh_path'])
    npy_u = mesh.sample_points_uniform([npy])
    
    print('Sending Data to Model')
    pred=psdxa.predict(npy_u,verbose=1)
    
    np.save(outpath+'pred.npy',pred)
    pred_resize=cv2.resize(pred[0],dsize=None,fx=3/2,fy=5/2, interpolation = cv2.INTER_LINEAR)
    out_raw = pred_resize[:,:,0]
    min_val = np.min(out_raw)
    max_val = np.max(out_raw)
    scaled_image = ((out_raw - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    cv2.imwrite(outpath+'raw_prediction.png', scaled_image)

    out_processed = pred_resize[:,:,3]-pred_resize[:,:,2]
    min_val = np.min(out_processed)
    max_val = np.max(out_processed)
    new_min, new_max = 0, 255
    out_processed = ((out_processed - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
    cv2.imwrite(outpath+'processed_prediction.png', out_processed.astype(np.uint8))
    
if __name__ == '__main__':
    main()