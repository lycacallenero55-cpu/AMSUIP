"""
CPU Optimization for TensorFlow
Optimized for Ryzen 5 3400G and similar CPUs
"""

import os
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

def configure_tensorflow_for_cpu():
    """
    Configure TensorFlow for optimal CPU performance
    """
    try:
        # Set CPU optimization flags
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
        os.environ['OMP_NUM_THREADS'] = '6'  # Match CPU cores
        os.environ['MKL_NUM_THREADS'] = '6'
        os.environ['NUMEXPR_NUM_THREADS'] = '6'
        
        # Configure TensorFlow for CPU
        tf.config.threading.set_inter_op_parallelism_threads(6)
        tf.config.threading.set_intra_op_parallelism_threads(6)
        
        # Enable CPU optimizations
        tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
        tf.config.optimizer.set_experimental_options({
            'layout_optimizer': True,
            'constant_folding': True,
            'shape_optimization': True,
            'remapping': True,
            'arithmetic_optimization': True,
            'dependency_optimization': True,
            'loop_optimization': True,
            'function_optimization': True,
            'debug_stripper': True,
            'scoped_allocator_optimization': True,
            'pin_to_host_optimization': True,
            'implementation_selector': True,
            'auto_mixed_precision': False,  # Disable for CPU
        })
        
        # Set memory growth to avoid allocating all GPU memory
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        
        logger.info("TensorFlow CPU optimization configured successfully")
        
    except Exception as e:
        logger.warning(f"CPU optimization configuration failed: {e}")
        # Continue without optimization