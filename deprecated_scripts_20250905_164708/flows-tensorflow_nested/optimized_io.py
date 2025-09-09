#!/usr/bin/env python3
"""
Optimized I/O utilities for TensorFlow Probability flows
Based on TensorFlow best practices for different use cases
"""

import numpy as np
import tensorflow as tf
import h5py
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json

def save_tensor_numpy(tensor: tf.Tensor, filepath: str, compress: bool = True) -> None:
    """Save single TensorFlow tensor using NumPy (most compatible).
    
    Args:
        tensor: TensorFlow tensor to save
        filepath: Output file path (.npy or .npz)
        compress: Whether to use compression
    """
    numpy_array = tensor.numpy()
    
    if compress or filepath.endswith('.npz'):
        np.savez_compressed(filepath, data=numpy_array)
    else:
        np.save(filepath, numpy_array)
    
    print(f"✅ Saved tensor {tensor.shape} to {filepath}")

def load_tensor_numpy(filepath: str, as_tensor: bool = True) -> Union[tf.Tensor, np.ndarray]:
    """Load tensor from NumPy file.
    
    Args:
        filepath: Input file path
        as_tensor: Whether to return as TensorFlow tensor
        
    Returns:
        TensorFlow tensor or NumPy array
    """
    if filepath.endswith('.npz'):
        data = np.load(filepath, allow_pickle=True)['data']
    else:
        data = np.load(filepath, allow_pickle=True)
    
    if as_tensor:
        return tf.constant(data, dtype=tf.float32)
    return data

def save_large_samples_hdf5(samples: tf.Tensor, masses: Optional[np.ndarray], 
                           filepath: str, metadata: Dict[str, Any]) -> None:
    """Save large sample datasets using HDF5 (optimal for large data).
    
    Args:
        samples: TensorFlow tensor of samples [N, 6] (pos + vel)
        masses: Optional array of stellar masses [N]
        filepath: Output HDF5 file path
        metadata: Dictionary of metadata to save as attributes
    """
    # Convert tensor to numpy for HDF5 compatibility
    samples_np = samples.numpy()
    
    with h5py.File(filepath, 'w') as f:
        # Main datasets with compression
        f.create_dataset('samples_6d', data=samples_np, compression='gzip', compression_opts=6)
        f.create_dataset('pos3', data=samples_np[:, :3], compression='gzip', compression_opts=6)
        f.create_dataset('vel3', data=samples_np[:, 3:], compression='gzip', compression_opts=6)
        
        # Masses if available
        if masses is not None:
            f.create_dataset('masses', data=masses, compression='gzip', compression_opts=6)
            f.attrs['has_masses'] = True
            f.attrs['total_mass'] = float(np.sum(masses))
            f.attrs['mean_mass'] = float(np.mean(masses))
        else:
            f.attrs['has_masses'] = False
        
        # Sample statistics (computed once, stored for quick access)
        f.attrs['n_samples'] = samples_np.shape[0]
        f.attrs['samples_mean'] = samples_np.mean(axis=0)
        f.attrs['samples_std'] = samples_np.std(axis=0)
        f.attrs['samples_min'] = samples_np.min(axis=0)
        f.attrs['samples_max'] = samples_np.max(axis=0)
        
        # Metadata
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                f.attrs[key] = value
            elif isinstance(value, (list, tuple, np.ndarray)):
                f.attrs[key] = np.array(value)
            else:
                f.attrs[f'{key}_json'] = json.dumps(value)
    
    print(f"✅ Saved {samples_np.shape[0]:,} samples to HDF5: {filepath}")

def load_large_samples_hdf5(filepath: str, load_as_tensors: bool = True) -> Dict[str, Any]:
    """Load large sample datasets from HDF5.
    
    Args:
        filepath: Input HDF5 file path
        load_as_tensors: Whether to convert arrays to TensorFlow tensors
        
    Returns:
        Dictionary containing samples, masses (if available), and metadata
    """
    result = {}
    
    with h5py.File(filepath, 'r') as f:
        # Load main datasets
        samples_6d = f['samples_6d'][:]
        pos3 = f['pos3'][:]
        vel3 = f['vel3'][:]
        
        if load_as_tensors:
            result['samples_6d'] = tf.constant(samples_6d, dtype=tf.float32)
            result['pos3'] = tf.constant(pos3, dtype=tf.float32)
            result['vel3'] = tf.constant(vel3, dtype=tf.float32)
        else:
            result['samples_6d'] = samples_6d
            result['pos3'] = pos3
            result['vel3'] = vel3
        
        # Load masses if available
        if f.attrs.get('has_masses', False):
            masses = f['masses'][:]
            result['masses'] = masses  # Keep as numpy for masses
        
        # Load metadata
        result['metadata'] = {}
        for key, value in f.attrs.items():
            if key.endswith('_json'):
                # Deserialize JSON values
                result['metadata'][key[:-5]] = json.loads(value)
            else:
                result['metadata'][key] = value
    
    print(f"✅ Loaded {result['samples_6d'].shape[0]:,} samples from HDF5: {filepath}")
    return result

def save_flow_weights_keras(flow_model, filepath: str) -> None:
    """Save TensorFlow flow model weights using Keras API (standard for models).
    
    Args:
        flow_model: TensorFlow/Keras model
        filepath: Output file path (.h5 or .weights.h5)
    """
    flow_model.save_weights(filepath)
    print(f"✅ Saved model weights to: {filepath}")

def serialize_tensor_tf(tensor: tf.Tensor) -> bytes:
    """Serialize tensor using TensorFlow's native serialization (for embedding).
    
    Args:
        tensor: TensorFlow tensor to serialize
        
    Returns:
        Serialized tensor as bytes
    """
    return tf.io.serialize_tensor(tensor).numpy()

def deserialize_tensor_tf(serialized_data: bytes) -> tf.Tensor:
    """Deserialize tensor using TensorFlow's native deserialization.
    
    Args:
        serialized_data: Serialized tensor bytes
        
    Returns:
        TensorFlow tensor
    """
    return tf.io.parse_tensor(serialized_data, out_type=tf.float32)

def save_samples_optimized(samples: tf.Tensor, masses: Optional[np.ndarray],
                          output_dir: str, model_name: str, metadata: Dict[str, Any]) -> Dict[str, str]:
    """Optimized save strategy using best practices for different formats.
    
    Args:
        samples: TensorFlow tensor of samples
        masses: Optional stellar masses
        output_dir: Output directory
        model_name: Base name for files
        metadata: Metadata dictionary
        
    Returns:
        Dictionary of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    n_samples = samples.shape[0]
    
    # Strategy based on sample count and use case
    if n_samples > 1_000_000:
        # Large datasets: Use HDF5 for optimal performance
        hdf5_path = output_path / f"{model_name}_samples.h5"
        save_large_samples_hdf5(samples, masses, str(hdf5_path), metadata)
        saved_files['hdf5'] = str(hdf5_path)
        
        # Also save a compressed NumPy version for quick loading
        npz_path = output_path / f"{model_name}_samples_quick.npz"
        quick_data = {
            'samples_6d': samples.numpy(),
            'pos3': samples[:, :3].numpy(),
            'vel3': samples[:, 3:].numpy()
        }
        if masses is not None:
            quick_data['masses'] = masses
        np.savez_compressed(npz_path, **quick_data)
        saved_files['npz_quick'] = str(npz_path)
        
    else:
        # Smaller datasets: NumPy is fine and more portable
        npz_path = output_path / f"{model_name}_samples.npz"
        data_dict = {
            'samples_6d': samples.numpy(),
            'pos3': samples[:, :3].numpy(),
            'vel3': samples[:, 3:].numpy(),
            'metadata': metadata
        }
        if masses is not None:
            data_dict['masses'] = masses
        
        np.savez_compressed(npz_path, **data_dict)
        saved_files['npz'] = str(npz_path)
    
    # Always save metadata as JSON for easy inspection
    metadata_path = output_path / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        # Convert any numpy types to Python types for JSON
        json_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                json_metadata[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_metadata[key] = value.item()
            else:
                json_metadata[key] = value
        json.dump(json_metadata, f, indent=2)
    saved_files['metadata'] = str(metadata_path)
    
    print(f"✅ Optimized save complete for {n_samples:,} samples")
    print(f"   Strategy: {'HDF5 + NPZ' if n_samples > 1_000_000 else 'NPZ only'}")
    print(f"   Files: {list(saved_files.keys())}")
    
    return saved_files

def load_samples_optimized(file_path: str, load_as_tensors: bool = True) -> Dict[str, Any]:
    """Load samples using the appropriate method based on file type.
    
    Args:
        file_path: Path to samples file (.h5, .npz, etc.)
        load_as_tensors: Whether to convert to TensorFlow tensors
        
    Returns:
        Dictionary containing loaded data
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.h5':
        return load_large_samples_hdf5(str(file_path), load_as_tensors)
    elif file_path.suffix == '.npz':
        data = np.load(file_path, allow_pickle=True)
        result = {}
        
        for key in data.files:
            array_data = data[key]
            if load_as_tensors and key in ['samples_6d', 'pos3', 'vel3']:
                result[key] = tf.constant(array_data, dtype=tf.float32)
            else:
                result[key] = array_data
        
        print(f"✅ Loaded samples from NPZ: {file_path}")
        return result
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

# Test function
def test_io_methods():
    """Test all I/O methods with sample data."""
    print("🧪 Testing optimized I/O methods...")
    
    # Create test data
    samples = tf.random.normal((50000, 6))  # 50k samples
    masses = np.random.lognormal(0, 1, 50000)  # Log-normal masses
    metadata = {
        'halo_id': '023',
        'particle_pid': 1,
        'n_training_epochs': 100,
        'created_date': '2024-08-23'
    }
    
    # Test optimized save
    saved_files = save_samples_optimized(
        samples=samples,
        masses=masses,
        output_dir='test_io_output',
        model_name='test_flow',
        metadata=metadata
    )
    
    # Test loading
    for file_type, file_path in saved_files.items():
        if file_type in ['hdf5', 'npz']:
            loaded_data = load_samples_optimized(file_path)
            print(f"✅ Successfully loaded {file_type}: {loaded_data['samples_6d'].shape}")
    
    # Cleanup
    import shutil
    shutil.rmtree('test_io_output')
    print("✅ I/O test completed and cleaned up")

if __name__ == "__main__":
    test_io_methods()
