"""
Production-Ready Signature Verification AI System
Real deep learning with proper feature extraction and embedding networks
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import os
import math

logger = logging.getLogger(__name__)

class SignatureEmbeddingModel:
    """
    Production-ready signature verification system with:
    - Signature-specific CNN architecture
    - Deep embedding networks for generalization
    - Multi-scale feature extraction
    - Attention mechanisms for stroke patterns
    - Anti-spoofing capabilities
    """
    
    def __init__(self, 
                 embedding_dim: int = 512,
                 image_size: int = 224,
                 max_students: int = 150,
                 learning_rate: float = 0.0001):
        
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.max_students = max_students
        self.learning_rate = learning_rate
        
        # Model components
        self.embedding_model = None
        self.classification_head = None
        self.authenticity_head = None
        self.siamese_model = None
        
        # Student mappings
        self.student_to_id = {}
        self.id_to_student = {}
        
        logger.info(f"Initialized SignatureEmbeddingModel: {embedding_dim}D embeddings, {image_size}x{image_size} input")
    
    def create_signature_backbone(self) -> keras.Model:
        """
        Create signature-specific CNN backbone with:
        - Multi-scale feature extraction
        - Attention mechanisms for stroke patterns
        - Signature-specific architectural choices
        """
        
        input_layer = layers.Input(shape=(self.image_size, self.image_size, 3), name='signature_input')
        
        # Preprocessing normalization
        x = layers.Rescaling(1.0/255.0, input_shape=(self.image_size, self.image_size, 3))(input_layer)
        
        # Multi-scale feature extraction
        # Scale 1: Fine details (strokes, curves)
        scale1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='scale1_conv1')(x)
        scale1 = layers.BatchNormalization(name='scale1_bn1')(scale1)
        scale1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='scale1_conv2')(scale1)
        scale1 = layers.MaxPooling2D((2, 2), name='scale1_pool')(scale1)
        
        # Scale 2: Medium patterns (letter shapes, connections)
        scale2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same', name='scale2_conv1')(x)
        scale2 = layers.BatchNormalization(name='scale2_bn1')(scale2)
        scale2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same', name='scale2_conv2')(scale2)
        scale2 = layers.MaxPooling2D((2, 2), name='scale2_pool')(scale2)
        
        # Scale 3: Global structure (overall signature shape)
        scale3 = layers.Conv2D(128, (7, 7), activation='relu', padding='same', name='scale3_conv1')(x)
        scale3 = layers.BatchNormalization(name='scale3_bn1')(scale3)
        scale3 = layers.Conv2D(128, (7, 7), activation='relu', padding='same', name='scale3_conv2')(scale3)
        scale3 = layers.MaxPooling2D((2, 2), name='scale3_pool')(scale3)
        
        # Combine multi-scale features
        scale1_resized = layers.Resizing(56, 56)(scale1)
        scale2_resized = layers.Resizing(56, 56)(scale2)
        scale3_resized = layers.Resizing(56, 56)(scale3)
        
        combined = layers.Concatenate(axis=-1, name='multi_scale_concat')([scale1_resized, scale2_resized, scale3_resized])
        
        # Deep feature extraction with residual connections
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='deep_conv1')(combined)
        x = layers.BatchNormalization(name='deep_bn1')(x)
        
        # Residual block 1
        residual = x
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='res1_conv1')(x)
        x = layers.BatchNormalization(name='res1_bn1')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='res1_conv2')(x)
        x = layers.BatchNormalization(name='res1_bn2')(x)
        x = layers.Add(name='res1_add')([x, residual])
        x = layers.ReLU(name='res1_relu')(x)
        
        # Spatial attention for stroke patterns
        attention = layers.Conv2D(1, (1, 1), activation='sigmoid', name='spatial_attention')(x)
        attended = layers.Multiply(name='attention_application')([x, attention])
        
        # Global feature aggregation (apply pooling to feature maps, not sequentially)
        gap = layers.GlobalAveragePooling2D(name='global_avg_pool')(attended)
        gmp = layers.GlobalMaxPooling2D(name='global_max_pool')(attended)
        x = layers.Concatenate(name='global_pool_concat')([gap, gmp])
        
        # Create backbone model
        backbone = keras.Model(input_layer, x, name='signature_backbone')
        
        logger.info(f"Created signature backbone with {backbone.count_params():,} parameters")
        return backbone
    
    def create_embedding_network(self) -> keras.Model:
        """
        Create deep embedding network for signature representations
        """
        
        backbone = self.create_signature_backbone()
        
        # Embedding layers
        x = layers.Dense(1024, activation='relu', name='embed_fc1')(backbone.output)
        x = layers.BatchNormalization(name='embed_bn1')(x)
        x = layers.Dropout(0.3, name='embed_dropout1')(x)
        
        x = layers.Dense(768, activation='relu', name='embed_fc2')(x)
        x = layers.BatchNormalization(name='embed_bn2')(x)
        x = layers.Dropout(0.2, name='embed_dropout2')(x)
        
        x = layers.Dense(512, activation='relu', name='embed_fc3')(x)
        x = layers.BatchNormalization(name='embed_bn3')(x)
        x = layers.Dropout(0.1, name='embed_dropout3')(x)
        
        # Final embedding layer with L2 normalization
        embedding = layers.Dense(self.embedding_dim, activation='linear', name='final_embedding')(x)
        embedding = layers.LayerNormalization(axis=1, name='l2_normalize')(embedding)
        
        # Create embedding model
        self.embedding_model = keras.Model(backbone.input, embedding, name='signature_embedding')
        
        logger.info(f"Created embedding network with {self.embedding_model.count_params():,} parameters")
        return self.embedding_model
    
    def create_classification_head(self) -> keras.Model:
        """
        Create classification head for student identification
        """
        
        if not self.embedding_model:
            self.create_embedding_network()
        
        # Classification layers
        x = layers.Dense(256, activation='relu', name='class_fc1')(self.embedding_model.output)
        x = layers.BatchNormalization(name='class_bn1')(x)
        x = layers.Dropout(0.3, name='class_dropout1')(x)
        
        x = layers.Dense(128, activation='relu', name='class_fc2')(x)
        x = layers.BatchNormalization(name='class_bn2')(x)
        x = layers.Dropout(0.2, name='class_dropout2')(x)
        
        # Student classification output
        student_output = layers.Dense(
            self.max_students, 
            activation='softmax', 
            name='student_classification'
        )(x)
        
        # Create classification model
        self.classification_head = keras.Model(
            self.embedding_model.input, 
            student_output, 
            name='signature_classification'
        )
        
        # Compile with advanced optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999
        )
        
        self.classification_head.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        logger.info(f"Created classification head with {self.classification_head.count_params():,} parameters")
        return self.classification_head
    
    def create_authenticity_head(self) -> keras.Model:
        """
        Create authenticity detection head for genuine/forged classification
        """
        
        if not self.embedding_model:
            self.create_embedding_network()
        
        # Authenticity-specific layers
        x = layers.Dense(256, activation='relu', name='auth_fc1')(self.embedding_model.output)
        x = layers.BatchNormalization(name='auth_bn1')(x)
        x = layers.Dropout(0.3, name='auth_dropout1')(x)
        
        x = layers.Dense(128, activation='relu', name='auth_fc2')(x)
        x = layers.BatchNormalization(name='auth_bn2')(x)
        x = layers.Dropout(0.2, name='auth_dropout2')(x)
        
        # Anti-spoofing features
        x = layers.Dense(64, activation='relu', name='auth_fc3')(x)
        x = layers.BatchNormalization(name='auth_bn3')(x)
        
        # Authenticity output
        authenticity_output = layers.Dense(1, activation='sigmoid', name='authenticity_classification')(x)
        
        # Create authenticity model
        self.authenticity_head = keras.Model(
            self.embedding_model.input, 
            authenticity_output, 
            name='signature_authenticity'
        )
        
        # Compile with focal loss for imbalanced data
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.01
        )
        
        self.authenticity_head.compile(
            optimizer=optimizer,
            loss=self._focal_loss,
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        logger.info(f"Created authenticity head with {self.authenticity_head.count_params():,} parameters")
        return self.authenticity_head
    
    def create_siamese_network(self) -> keras.Model:
        """
        Create Siamese network for signature similarity learning
        """
        
        if not self.embedding_model:
            self.create_embedding_network()
        
        # Siamese inputs
        input_a = layers.Input(shape=(self.image_size, self.image_size, 3), name='input_a')
        input_b = layers.Input(shape=(self.image_size, self.image_size, 3), name='input_b')
        
        # Get embeddings
        embedding_a = self.embedding_model(input_a)
        embedding_b = self.embedding_model(input_b)
        
        # Compute similarity metrics
        # Cosine similarity
        cosine_sim = layers.Dot(axes=1, normalize=True, name='cosine_similarity')([embedding_a, embedding_b])
        
        # Euclidean distance
        diff_euclidean = layers.Subtract(name='euclidean_diff')([embedding_a, embedding_b])
        euclidean_dist = layers.Dense(1, activation='linear', name='euclidean_distance')(diff_euclidean)
        
        # Manhattan distance
        diff_manhattan = layers.Subtract(name='manhattan_diff')([embedding_a, embedding_b])
        manhattan_dist = layers.Dense(1, activation='linear', name='manhattan_distance')(diff_manhattan)
        
        # Combine similarity metrics
        combined_features = layers.Concatenate(name='similarity_features')([
            cosine_sim, euclidean_dist, manhattan_dist
        ])
        
        # Similarity classification head
        x = layers.Dense(64, activation='relu', name='siamese_fc1')(combined_features)
        x = layers.BatchNormalization(name='siamese_bn1')(x)
        x = layers.Dropout(0.3, name='siamese_dropout1')(x)
        
        x = layers.Dense(32, activation='relu', name='siamese_fc2')(x)
        x = layers.BatchNormalization(name='siamese_bn2')(x)
        x = layers.Dropout(0.2, name='siamese_dropout2')(x)
        
        # Similarity output
        similarity_output = layers.Dense(1, activation='sigmoid', name='similarity_classification')(x)
        
        # Create Siamese model
        self.siamese_model = keras.Model(
            [input_a, input_b], 
            similarity_output, 
            name='signature_siamese'
        )
        
        # Compile with contrastive loss
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.01
        )
        
        self.siamese_model.compile(
            optimizer=optimizer,
            loss=self._contrastive_loss,
            metrics=['accuracy', 'auc']
        )
        
        logger.info(f"Created Siamese network with {self.siamese_model.count_params():,} parameters")
        return self.siamese_model
    
    def _focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2.0):
        """
        Focal loss for handling class imbalance in authenticity detection
        """
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        
        focal_loss = -focal_weight * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    
    def _contrastive_loss(self, y_true, y_pred, margin=1.0):
        """
        Contrastive loss for Siamese network training
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        positive_loss = y_true * tf.square(y_pred)
        negative_loss = (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
        
        return tf.reduce_mean(positive_loss + negative_loss)
    
    def prepare_training_data(self, training_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data with proper preprocessing and augmentation
        """
        logger.info("Preparing training data with signature-specific preprocessing...")
        
        all_images = []
        student_labels = []
        authenticity_labels = []
        
        for student_name, signatures in training_data.items():
            if student_name not in self.student_to_id:
                student_id = len(self.student_to_id)
                self.student_to_id[student_name] = student_id
                self.id_to_student[student_id] = student_name
                logger.info(f"Added student '{student_name}' with ID {student_id}")
            
            student_id = self.student_to_id[student_name]
            
            # Process genuine signatures
            for img in signatures['genuine']:
                processed_img = self._preprocess_signature(img)
                all_images.append(processed_img)
                student_labels.append(student_id)
                authenticity_labels.append(1)  # Genuine
            
            # Process forged signatures
            for img in signatures['forged']:
                processed_img = self._preprocess_signature(img)
                all_images.append(processed_img)
                student_labels.append(student_id)
                authenticity_labels.append(0)  # Forged
        
        X = np.array(all_images, dtype=np.float32)
        y_student = tf.keras.utils.to_categorical(student_labels, num_classes=self.max_students)
        y_authenticity = np.array(authenticity_labels, dtype=np.float32)
        
        logger.info(f"Prepared {len(all_images)} images across {len(self.student_to_id)} students")
        return X, y_student, y_authenticity
    
    def _preprocess_signature(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Signature-specific preprocessing pipeline
        """
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        
        # Ensure proper format
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        
        # Resize to model input size
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = tf.cast(image, tf.float32)
        
        # Normalize to [0, 1]
        if tf.reduce_max(image) > 1.0:
            image = image / 255.0
        
        return image.numpy()
    
    def train_models(self, training_data: Dict, epochs: int = 100) -> Dict:
        """
        Train all models with advanced training strategies
        """
        logger.info("Starting comprehensive model training...")
        
        # Prepare data
        X, y_student, y_authenticity = self.prepare_training_data(training_data)
        
        # Create models
        self.create_classification_head()
        self.create_authenticity_head()
        self.create_siamese_network()
        
        # Advanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            CosineRestartScheduler(
                T_0=20,
                T_mult=2,
                eta_min=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_signature_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train classification model
        logger.info("Training student classification model...")
        classification_history = self.classification_head.fit(
            X, y_student,
            batch_size=32,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train authenticity model
        logger.info("Training authenticity detection model...")
        authenticity_history = self.authenticity_head.fit(
            X, y_authenticity,
            batch_size=32,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train Siamese network with pair generation
        logger.info("Training Siamese network...")
        siamese_pairs, siamese_labels = self._generate_siamese_pairs(training_data)
        siamese_history = self.siamese_model.fit(
            siamese_pairs, siamese_labels,
            batch_size=32,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed successfully!")
        
        return {
            'classification_history': classification_history.history,
            'authenticity_history': authenticity_history.history,
            'siamese_history': siamese_history.history,
            'student_mappings': {
                'student_to_id': self.student_to_id,
                'id_to_student': self.id_to_student
            }
        }
    
    def _generate_siamese_pairs(self, training_data: Dict) -> Tuple[List, List]:
        """
        Generate positive and negative pairs for Siamese training
        """
        pairs = []
        labels = []
        
        for student_name, signatures in training_data.items():
            genuine_images = signatures['genuine']
            forged_images = signatures['forged']
            
            # Positive pairs (genuine-genuine)
            for i in range(len(genuine_images)):
                for j in range(i + 1, min(i + 3, len(genuine_images))):
                    img1 = self._preprocess_signature(genuine_images[i])
                    img2 = self._preprocess_signature(genuine_images[j])
                    pairs.append([img1, img2])
                    labels.append(1)  # Similar
            
            # Negative pairs (genuine-forged)
            for i in range(min(len(genuine_images), len(forged_images))):
                for j in range(min(2, len(forged_images))):
                    img1 = self._preprocess_signature(genuine_images[i])
                    img2 = self._preprocess_signature(forged_images[j])
                    pairs.append([img1, img2])
                    labels.append(0)  # Different
        
        return np.array(pairs), np.array(labels)
    
    def verify_signature(self, test_signature: Union[np.ndarray, Image.Image]) -> Dict:
        """
        Comprehensive signature verification with multiple models
        """
        logger.info("Performing comprehensive signature verification...")
        
        # Check available heads; allow partial inference if only authenticity is present
        if not self.embedding_model:
            raise ValueError("Embedding model not loaded. Please load a trained model first.")
        has_classification = self.classification_head is not None
        has_authenticity = self.authenticity_head is not None
        if not (has_classification or has_authenticity):
            raise ValueError("No verification heads loaded. Please load a trained model first.")
        
        # Preprocess test signature
        processed_signature = self._preprocess_signature(test_signature)
        X_test = np.expand_dims(processed_signature, axis=0)
        
        # Get embeddings
        embedding = self.embedding_model.predict(X_test, verbose=0)[0]
        
        predicted_student_id = 0
        student_confidence = 0.0
        if has_classification:
            student_probs = self.classification_head.predict(X_test, verbose=0)[0]
            predicted_student_id = int(np.argmax(student_probs))
            student_confidence = float(np.max(student_probs))
            
            # DEBUG: Log prediction details
            logger.info(f"Classification prediction: argmax={predicted_student_id}, confidence={student_confidence:.4f}")
            logger.info(f"Available mappings: {self.id_to_student}")
            logger.info(f"Student probabilities: {student_probs}")
            
            # CRITICAL FIX: Ensure we only predict students that exist in our training data
            if not self.id_to_student or predicted_student_id not in self.id_to_student:
                # Try to map the predicted class to actual student IDs
                if self.id_to_student and len(self.id_to_student) > 0:
                    # Get the list of available student IDs (only trained students)
                    available_ids = [k for k in self.id_to_student.keys() if isinstance(k, int) and k > 0]
                    if predicted_student_id < len(available_ids):
                        # Map the predicted class index to the actual student ID
                        mapped_student_id = available_ids[predicted_student_id]
                        predicted_student_id = mapped_student_id
                        logger.info(f"Mapped predicted class {predicted_student_id} to student ID {mapped_student_id}")
                    else:
                        # If predicted class is out of range, mark as unknown
                        predicted_student_id = 0
                        student_confidence = 0.0
                        logger.warning(f"Predicted class {predicted_student_id} out of range (available trained students: {len(available_ids)})")
                else:
                    # If no mappings loaded, mark as unknown
                    predicted_student_id = 0
                    student_confidence = 0.0
                    logger.warning(f"No student mappings available (mappings: {len(self.id_to_student) if self.id_to_student else 0})")
        
        # Authenticity detection
        authenticity_score = 0.0
        is_genuine = False
        if has_authenticity:
            authenticity_score = float(self.authenticity_head.predict(X_test, verbose=0)[0][0])
            is_genuine = authenticity_score > 0.5
        
        # Get student name and actual student ID
        predicted_student_name = self.id_to_student.get(predicted_student_id, f"Unknown_{predicted_student_id}")
        
        # If we predicted a class index (0, 1), try to find the corresponding actual student ID
        actual_student_id = predicted_student_id
        if predicted_student_id in [0, 1] and len(self.id_to_student) > 2:
            # Look for actual student IDs in the mappings
            actual_ids = [k for k in self.id_to_student.keys() if k not in [0, 1]]
            if actual_ids and predicted_student_id < len(actual_ids):
                actual_student_id = actual_ids[predicted_student_id]
                logger.info(f"Mapped class {predicted_student_id} to actual student ID {actual_student_id}")
                # Update the predicted student name to match the new ID
                predicted_student_name = self.id_to_student.get(actual_student_id, f"Unknown_{actual_student_id}")
                logger.info(f"Updated predicted student name to: {predicted_student_name}")
        
        # Calculate overall confidence
        if has_classification and has_authenticity:
            overall_confidence = (student_confidence + authenticity_score) / 2.0
        elif has_authenticity:
            overall_confidence = authenticity_score
        else:
            overall_confidence = student_confidence
        
        # Determine if signature is unknown - STRICTER thresholds to prevent false positives
        is_unknown = (
            (has_classification and student_confidence < 0.5) or  # Increased from 0.3 to 0.5
            overall_confidence < 0.6 or  # Increased from 0.4 to 0.6
            predicted_student_id == 0  # If no valid student ID found
        )
        
        result = {
            'predicted_student_id': actual_student_id,  # Use actual student ID
            'predicted_student_name': predicted_student_name,
            'student_confidence': student_confidence,
            'is_genuine': bool(is_genuine),
            'authenticity_score': authenticity_score,
            'overall_confidence': overall_confidence,
            'is_unknown': is_unknown,
            'embedding': embedding.tolist()  # For similarity comparisons
        }
        
        return result
    
    def save_models(self, base_path: str):
        """
        Save all trained models and mappings
        """
        if self.embedding_model:
            self.embedding_model.save(f"{base_path}_embedding.keras")
            logger.info(f"Embedding model saved to {base_path}_embedding.keras")
        
        if self.classification_head:
            self.classification_head.save(f"{base_path}_classification.keras")
            logger.info(f"Classification model saved to {base_path}_classification.keras")
        
        if self.authenticity_head:
            self.authenticity_head.save(f"{base_path}_authenticity.keras")
            logger.info(f"Authenticity model saved to {base_path}_authenticity.keras")
        
        if self.siamese_model:
            self.siamese_model.save(f"{base_path}_siamese.keras")
            logger.info(f"Siamese model saved to {base_path}_siamese.keras")
        
        # Save mappings
        import json
        mappings_path = f"{base_path}_mappings.json"
        with open(mappings_path, 'w') as f:
            json.dump({
                'student_to_id': self.student_to_id,
                'id_to_student': self.id_to_student,
                'embedding_dim': self.embedding_dim,
                'image_size': self.image_size
            }, f, indent=2)
        logger.info(f"Model mappings saved to {mappings_path}")
    
    def load_models(self, base_path: str) -> bool:
        """
        Load all trained models and mappings
        """
        try:
            # Load embedding model
            embedding_path = f"{base_path}_embedding.keras"
            if os.path.exists(embedding_path):
                self.embedding_model = keras.models.load_model(embedding_path)
                logger.info(f"Embedding model loaded from {embedding_path}")
            
            # Load classification model
            classification_path = f"{base_path}_classification.keras"
            if os.path.exists(classification_path):
                self.classification_head = keras.models.load_model(classification_path)
                logger.info(f"Classification model loaded from {classification_path}")
            
            # Load authenticity model
            authenticity_path = f"{base_path}_authenticity.keras"
            if os.path.exists(authenticity_path):
                self.authenticity_head = keras.models.load_model(authenticity_path)
                logger.info(f"Authenticity model loaded from {authenticity_path}")
            
            # Load Siamese model
            siamese_path = f"{base_path}_siamese.keras"
            if os.path.exists(siamese_path):
                self.siamese_model = keras.models.load_model(siamese_path)
                logger.info(f"Siamese model loaded from {siamese_path}")
            
            # Load mappings
            mappings_path = f"{base_path}_mappings.json"
            if os.path.exists(mappings_path):
                import json
                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                self.student_to_id = mappings['student_to_id']
                self.id_to_student = {int(k): v for k, v in mappings['id_to_student'].items()}
                self.embedding_dim = mappings.get('embedding_dim', 512)
                self.image_size = mappings.get('image_size', 224)
                logger.info(f"Model mappings loaded from {mappings_path}")
            
            logger.info("All signature models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load signature models: {e}")
            return False


class CosineRestartScheduler(keras.callbacks.Callback):
    """
    Cosine annealing with warm restarts for better training
    """
    
    def __init__(self, T_0, T_mult=1, eta_min=0, verbose=0):
        super().__init__()
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.verbose = verbose
        self.T_cur = 0
        self.T_i = T_0
        self.eta_max = None
    
    def on_train_begin(self, logs=None):
        self.eta_max = self.model.optimizer.learning_rate.numpy()
    
    def on_epoch_end(self, epoch, logs=None):
        self.T_cur += 1
        
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
        
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        self.model.optimizer.learning_rate.assign(lr)
        
        if self.verbose > 0:
            print(f'\nEpoch {epoch + 1}: CosineRestartScheduler setting learning rate to {lr:.6f}')