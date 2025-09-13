# global_signature_model.py - Global Multi-Student Signature Verification Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from config import settings
import logging
from typing import List, Tuple, Optional, Union, Dict
from PIL import Image
import os

logger = logging.getLogger(__name__)

class GlobalSignatureVerificationModel:
    """Global embedding model for multi-student signature verification"""
    
    def __init__(self):
        self.model = None
        self.embedding_model = None
        self.image_size = settings.MODEL_IMAGE_SIZE
        self.batch_size = settings.MODEL_BATCH_SIZE
        self.epochs = settings.MODEL_EPOCHS
        self.learning_rate = settings.MODEL_LEARNING_RATE
        self.embedding_dim = 128  # Global embedding dimension
        
    def create_global_embedding_model(self):
        """Create a global embedding model for all students"""
        
        # Input layer
        input_layer = layers.Input(shape=(self.image_size, self.image_size, 3), name='input')
        
        # Base model (MobileNetV2)
        base = keras.applications.MobileNetV2(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze most layers to prevent overfitting
        for layer in base.layers[:-30]:  # Unfreeze last 30 layers
            layer.trainable = False
        
        # Preprocessing
        x = keras.applications.mobilenet_v2.preprocess_input(input_layer)
        x = base(x, training=False)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Feature extraction layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Final embedding layer
        x = layers.Dense(self.embedding_dim, activation='linear')(x)
        x = layers.LayerNormalization()(x)
        
        # L2 normalization for cosine similarity
        x = tf.nn.l2_normalize(x, axis=1)
        
        # Create model
        self.embedding_model = keras.Model(input_layer, x, name='global_embedding_model')
        
        logger.info(f"Created global embedding model with {self.embedding_model.count_params()} parameters")
        return self.embedding_model
    
    def create_siamese_network(self):
        """Create Siamese network for training"""
        
        # Create embedding model
        embedding_model = self.create_global_embedding_model()
        
        # Siamese inputs
        input_a = layers.Input(shape=(self.image_size, self.image_size, 3), name='input_a')
        input_b = layers.Input(shape=(self.image_size, self.image_size, 3), name='input_b')
        
        # Get embeddings
        embedding_a = embedding_model(input_a)
        embedding_b = embedding_model(input_b)
        
        # Compute distance
        distance = tf.norm(embedding_a - embedding_b, axis=1, keepdims=True)
        
        # Classification head
        output = layers.Dense(64, activation='relu')(distance)
        output = layers.Dropout(0.3)(output)
        output = layers.Dense(32, activation='relu')(output)
        output = layers.Dropout(0.2)(output)
        output = layers.Dense(1, activation='sigmoid')(output)
        
        # Create Siamese model
        self.model = keras.Model([input_a, input_b], output, name='global_siamese_model')
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, weight_decay=1e-4)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
            ]
        )
        
        logger.info(f"Created global Siamese model with {self.model.count_params()} parameters")
        return self.model
    
    def train_global_model(self, all_student_data: Dict[int, Dict]):
        """
        Train global model on data from all students
        
        Args:
            all_student_data: Dict mapping student_id to {
                'genuine_images': List[np.ndarray],
                'forged_images': List[np.ndarray]
            }
        """
        
        # Create model
        self.create_siamese_network()
        
        # Prepare training data
        all_pairs = []
        all_labels = []
        
        for student_id, data in all_student_data.items():
            genuine_images = data['genuine_images']
            forged_images = data['forged_images']
            
            # Generate positive pairs (genuine-genuine)
            for i in range(len(genuine_images)):
                for j in range(i + 1, min(i + 3, len(genuine_images))):
                    all_pairs.append([genuine_images[i], genuine_images[j]])
                    all_labels.append(1)  # Similar
            
            # Skip negative pairs with forged signatures - not used for owner identification
            # (Forgery detection is disabled - focus on owner identification only)
            
            # Generate cross-student negative pairs
            for other_student_id, other_data in all_student_data.items():
                if other_student_id != student_id:
                    other_genuine = other_data['genuine_images']
                    for i in range(min(1, len(genuine_images))):
                        for j in range(min(1, len(other_genuine))):
                            all_pairs.append([genuine_images[i], other_genuine[j]])
                            all_labels.append(0)  # Different student
        
        # Convert to numpy arrays
        pairs_array = np.array(all_pairs)
        labels_array = np.array(all_labels)
        
        logger.info(f"Generated {len(all_pairs)} training pairs from {len(all_student_data)} students")
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_pairs, val_pairs, train_labels, val_labels = train_test_split(
            pairs_array, labels_array, test_size=0.2, random_state=42, stratify=labels_array
        )
        
        # Prepare training data
        train_input_a = train_pairs[:, 0]
        train_input_b = train_pairs[:, 1]
        val_input_a = val_pairs[:, 0]
        val_input_b = val_pairs[:, 1]
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                mode='min',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.LearningRateScheduler(
                lambda epoch: self.learning_rate * (0.9 ** epoch)
            )
        ]
        
        # Train model
        history = self.model.fit(
            [train_input_a, train_input_b], train_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=([val_input_a, val_input_b], val_labels),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def embed_images(self, images: List[np.ndarray]) -> np.ndarray:
        """Generate embeddings for a list of images"""
        if not images:
            return np.array([])
        
        # Ensure images are in correct format
        processed_images = []
        for img in images:
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if img.max() > 1.0:  # Normalize if needed
                img = img / 255.0
            processed_images.append(img)
        
        images_array = np.array(processed_images)
        embeddings = self.embedding_model.predict(images_array, verbose=0)
        return embeddings
    
    def compute_student_centroids(self, all_student_data: Dict[int, Dict]) -> Dict[int, np.ndarray]:
        """Compute centroid for each student's genuine signatures"""
        centroids = {}
        
        for student_id, data in all_student_data.items():
            genuine_images = data['genuine_images']
            if genuine_images:
                embeddings = self.embed_images(genuine_images)
                centroid = np.mean(embeddings, axis=0)
                centroids[student_id] = centroid
                logger.info(f"Computed centroid for student {student_id}: shape {centroid.shape}")
        
        return centroids
    
    def save_model(self, model_path: str):
        """Save the global embedding model"""
        if self.embedding_model:
            self.embedding_model.save(model_path)
            logger.info(f"Saved global embedding model to {model_path}")
    
    def load_model(self, model_path: str):
        """Load the global embedding model"""
        self.embedding_model = keras.models.load_model(model_path)
        logger.info(f"Loaded global embedding model from {model_path}")
    
    def verify_signature(self, test_image: np.ndarray, student_centroids: Dict[int, np.ndarray], 
                        thresholds: Dict[int, float]) -> Dict:
        """
        Verify a signature against all student centroids
        
        Returns:
            Dict with 'best_match', 'best_score', 'all_scores', 'is_genuine'
        """
        # Get embedding for test image
        test_embedding = self.embed_images([test_image])[0]
        
        # Compute distances to all centroids
        all_scores = {}
        best_score = 0.0
        best_match = None
        
        for student_id, centroid in student_centroids.items():
            # Use cosine similarity for better generalization to new signatures
            cosine_sim = np.dot(test_embedding, centroid) / (np.linalg.norm(test_embedding) * np.linalg.norm(centroid))
            # Convert to distance (lower is better, range 0-2)
            distance = 1 - cosine_sim
            threshold = thresholds.get(student_id, 0.3)  # Adjusted for cosine distance
            
            # Use sigmoid scoring with better parameters for cosine distance
            import math
            k = 3.0  # Less aggressive than euclidean
            score = 1.0 / (1.0 + math.exp(k * (distance - threshold)))
            
            all_scores[student_id] = {
                'distance': float(distance),
                'threshold': float(threshold),
                'score': float(score),
                'is_genuine': distance <= threshold
            }
            
            if score > best_score:
                best_score = score
                best_match = student_id
        
        return {
            'best_match': best_match,
            'best_score': best_score,
            'all_scores': all_scores,
            'is_genuine': all_scores[best_match]['is_genuine'] if best_match else False
        }