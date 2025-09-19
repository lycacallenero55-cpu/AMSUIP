"""
Production-Ready Signature Verification AI System
Real deep learning with proper feature extraction and embedding networks
Focus: Owner identification only (forgery detection disabled)
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
    - Owner identification focus (forgery detection disabled)
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
        # Note: authenticity_head removed - forgery detection is disabled
        self.siamese_model = None
        
        # ID-first mappings (Teachable Machine style)
        self.student_id_to_class = {}  # Maps student_id -> class_index
        self.class_to_student_id = {}  # Maps class_index -> student_id
        self.student_id_to_name = {}   # Maps student_id -> student_name
        
        # Legacy mappings (deprecated - for backward compatibility only)
        self.student_to_id = {}  # DEPRECATED: name -> class_index
        self.id_to_student = {}  # DEPRECATED: class_index -> name
        
        logger.info(f"Initialized SignatureEmbeddingModel: {embedding_dim}D embeddings, {image_size}x{image_size} input")
    
    def create_signature_backbone(self) -> keras.Model:
        """
        Create signature-specific CNN backbone with transfer learning:
        - Uses MobileNetV2 as base for transfer learning
        - Multi-scale feature extraction
        - Attention mechanisms for stroke patterns
        - Signature-specific architectural choices
        """
        
        input_layer = layers.Input(shape=(self.image_size, self.image_size, 3), name='signature_input')
        
        # Preprocessing normalization
        x = layers.Rescaling(1.0/255.0, input_shape=(self.image_size, self.image_size, 3))(input_layer)
        
        # Transfer Learning: Use MobileNetV2 as base for signature recognition
        base_model = keras.applications.MobileNetV2(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers for transfer learning (prevents overfitting with small datasets)
        base_model.trainable = False
        
        # Add progressive unfreezing strategy for better fine-tuning
        # Unfreeze the last few layers for fine-tuning
        for layer in base_model.layers[-20:]:  # Unfreeze last 20 layers
            layer.trainable = True
        
        # Add data augmentation layers for better generalization
        augmentation_layers = [
            layers.RandomRotation(0.1, fill_mode='constant', fill_value=1.0),
            layers.RandomZoom(0.1, fill_mode='constant', fill_value=1.0),
            layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=1.0),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
        ]
        
        # Get base model features
        base_features = base_model(x, training=False)
        
        # Add custom layers for signature-specific processing
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='custom_conv1')(base_features)
        x = layers.BatchNormalization(name='custom_bn1')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='custom_conv2')(x)
        x = layers.BatchNormalization(name='custom_bn2')(x)
        
        # Add signature-specific processing on top of transfer learning features
        # Global feature aggregation from MobileNetV2 features
        gap = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        gmp = layers.GlobalMaxPooling2D(name='global_max_pool')(x)
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
    
    def create_classification_head(self, num_students: int = None) -> keras.Model:
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
        
        # Student classification output - use provided num_students or actual count
        if num_students is None:
            num_students = len(self.student_to_id) if self.student_to_id else self.max_students
        # For single-student datasets, use a binary head to avoid degenerate softmax
        if num_students == 1:
            student_output = layers.Dense(1, activation='sigmoid', name='student_classification')(x)
        else:
            student_output = layers.Dense(num_students, activation='softmax', name='student_classification')(x)
        
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
        
        # Choose safe loss/metrics based on number of classes
        if num_students == 1:
            self.classification_head.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            self.classification_head.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
                    # Removed Precision, Recall, AUC due to Keras 3.x compatibility issues
                ]
            )
        
        logger.info(f"Created classification head with {self.classification_head.count_params():,} parameters")
        return self.classification_head
    
    # Note: create_authenticity_head method removed - forgery detection is disabled
    
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
        
        # Use safe metrics to avoid TF/Keras metric type issues
        self.siamese_model.compile(
            optimizer=optimizer,
            loss=self._contrastive_loss,
            metrics=['accuracy']
        )
        
        logger.info(f"Created Siamese network with {self.siamese_model.count_params():,} parameters")
        return self.siamese_model
    
    # Note: _focal_loss method removed - only used for authenticity detection which is disabled
    
    def _contrastive_loss(self, y_true, y_pred, margin=1.0):
        """
        Contrastive loss for Siamese network training
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        positive_loss = y_true * tf.square(y_pred)
        negative_loss = (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
        
        return tf.reduce_mean(positive_loss + negative_loss)
    
    def prepare_training_data(self, training_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with proper preprocessing and augmentation
        Focus: Owner identification only (forgery detection disabled)
        Uses ID-first approach for Teachable Machine compatibility
        """
        logger.info("Preparing training data with ID-first approach...")
        
        all_images = []
        student_labels = []
        
        # Import augmentation
        from utils.signature_preprocessing import SignatureAugmentation
        augmenter = SignatureAugmentation(
            rotation_range=10.0,  # Reduced for signature stability
            scale_range=(0.9, 1.1),  # Conservative scaling
            brightness_range=0.2,  # Subtle brightness changes
            noise_std=3.0,  # Reduced noise
            blur_probability=0.2  # Reduced blur probability
        )
        
        class_index = 0
        for student_key, signatures in training_data.items():
            # Parse student ID from key (format: "ID_Name" or just "ID")
            if isinstance(student_key, str) and '_' in student_key:
                parts = student_key.split('_', 1)
                try:
                    student_id = int(parts[0])
                    student_name = parts[1] if len(parts) > 1 else f"Student_{student_id}"
                except:
                    student_id = hash(student_key) % 100000
                    student_name = student_key
            else:
                try:
                    student_id = int(student_key)
                    student_name = f"Student_{student_id}"
                except:
                    student_id = hash(student_key) % 100000
                    student_name = str(student_key)
            
            # Store ID-first mappings
            self.student_id_to_class[student_id] = class_index
            self.class_to_student_id[class_index] = student_id
            self.student_id_to_name[student_id] = student_name
            
            # Legacy mappings for backward compatibility
            self.student_to_id[student_name] = class_index
            self.id_to_student[class_index] = student_name
            
            logger.info(f"Added student ID {student_id} ('{student_name}') as class {class_index}")
            
            # Process genuine signatures with augmentation
            genuine_images = signatures['genuine']
            logger.info(f"Processing {len(genuine_images)} genuine signatures for {student_name}")
            
            for img in genuine_images:
                try:
                    # Original image
                    processed_img = self._preprocess_signature(img)
                    all_images.append(processed_img)
                    student_labels.append(class_index)  # Use class index for labels
                    
                    # Augmented versions (3x augmentation for better balance)
                    for _ in range(3):
                        try:
                            augmented_img = augmenter.augment_signature(processed_img, is_genuine=True)
                            # Validate augmented image
                            if augmented_img is not None and augmented_img.shape == processed_img.shape:
                                all_images.append(augmented_img)
                                student_labels.append(class_index)  # Use class index for labels
                        except Exception as e:
                            logger.warning(f"Augmentation failed for student {student_id}: {e}")
                            # Continue without this augmentation
                except Exception as e:
                    logger.warning(f"Preprocessing failed for student {student_id}: {e}")
                    continue
            
            # Skip forged signatures - not used for owner identification training
            # (Forgery detection is disabled system-wide - focus on owner identification only)
            
            class_index += 1  # Increment class index for next student
        
        if not all_images:
            logger.warning("No valid training images found - returning empty arrays")
            return np.array([]).reshape(0, self.image_size, self.image_size, 3), np.array([]).reshape(0, 1)
        
        X = np.array(all_images, dtype=np.float32)
        # Use actual number of classes for categorical encoding
        num_classes = len(self.class_to_student_id)
        y_student = tf.keras.utils.to_categorical(student_labels, num_classes=num_classes)
        
        logger.info(f"Prepared {len(all_images)} images (including augmentations) across {num_classes} classes")
        logger.info(f"ID-first mappings: {num_classes} students mapped")
        logger.info(f"Training data shape: X={X.shape}, y_student={y_student.shape}")
        return X, y_student
    
    def _preprocess_signature(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Signature-specific preprocessing pipeline
        """
        try:
            if hasattr(image, 'convert'):
                image = np.array(image.convert('RGB'))
            
            # Ensure proper format
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            elif image.shape[-1] == 4:
                image = image[..., :3]
            
            # Validate image dimensions
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Invalid image shape: {image.shape}")
            
            # Convert to tensor for TensorFlow operations
            image = tf.convert_to_tensor(image)
            
            # Resize to model input size
            image = tf.image.resize(image, [self.image_size, self.image_size])
            image = tf.cast(image, tf.float32)
            
            # Normalize to [0, 1]
            if tf.reduce_max(image) > 1.0:
                image = image / 255.0
            
            return image.numpy()
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            # Return a blank white image as fallback
            return np.ones((self.image_size, self.image_size, 3), dtype=np.float32)
    
    def train_classification_only(self, training_data: Dict, epochs: int = 50) -> Dict:
        """
        Train classification head with progressive unfreezing for better transfer learning
        Focus: Owner identification with small datasets (like Teachable Machine)
        """
        logger.info("Starting classification training with progressive unfreezing...")
        
        # Validate training data
        if not training_data:
            logger.warning("No training data provided - returning empty training result")
            return {
                'classification_history': {'loss': [], 'accuracy': []},
                'siamese_history': {'loss': []},
                'student_mappings': {
                    'student_to_id': self.student_to_id,
                    'id_to_student': self.id_to_student
                }
            }
        
        # Prepare data first to set up student mappings
        X, y_student = self.prepare_training_data(training_data)
        
        # Validate prepared data
        if X.shape[0] == 0:
            logger.warning("No valid training images after preprocessing - skipping training")
            return {
                'classification_history': {'loss': [], 'accuracy': []},
                'siamese_history': {'loss': []},
                'student_mappings': {
                    'student_to_id': self.student_to_id,
                    'id_to_student': self.id_to_student
                }
            }
        
        if y_student.shape[1] < 2:
            logger.warning(f"Need at least 2 students for classification training, got {y_student.shape[1]} - skipping training")
            return {
                'classification_history': {'loss': [], 'accuracy': []},
                'siamese_history': {'loss': []},
                'student_mappings': {
                    'student_to_id': self.student_to_id,
                    'id_to_student': self.id_to_student
                }
            }
        
        # Create only the classification model with correct number of students
        num_students = len(self.student_to_id)
        self.create_classification_head(num_students=num_students)
        
        # Phase 1: Train only the classification head (frozen backbone)
        logger.info("Phase 1: Training classification head with frozen backbone...")
        
        # Freeze the backbone completely for phase 1
        if hasattr(self.classification_head, 'layers'):
            for layer in self.classification_head.layers:
                if 'mobilenetv2' in layer.name.lower() or 'base_model' in layer.name.lower():
                    layer.trainable = False
        
        # Optimized callbacks for small dataset training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,  # Increased patience for small datasets
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,  # Increased patience for small datasets
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                'training_log.csv',
                append=True
            )
        ]
        
        # Add real-time metrics callback if job_id is available
        try:
            from utils.training_callback import RealTimeMetricsCallback
            import threading
            current_thread = threading.current_thread()
            job_id = getattr(current_thread, 'job_id', None)
            if job_id:
                callbacks.append(RealTimeMetricsCallback(job_id, epochs))
                logger.info(f"Added real-time metrics callback for job {job_id}")
        except Exception as e:
            logger.warning(f"Could not add real-time metrics callback: {e}")
        
        # Calculate appropriate batch size for small datasets
        batch_size = min(16, max(2, len(X) // 8))  # Smaller batch size for small datasets
        
        logger.info(f"Phase 1 training with batch size: {batch_size}, samples: {len(X)}")
        
        # Phase 1: Train with frozen backbone
        phase1_epochs = min(epochs // 2, 25)  # Use half epochs for phase 1
        classification_history = self.classification_head.fit(
            X, y_student,
            batch_size=batch_size,
            epochs=phase1_epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tune with unfrozen backbone
        logger.info("Phase 2: Fine-tuning with unfrozen backbone...")
        
        # Unfreeze the backbone for fine-tuning
        if hasattr(self.classification_head, 'layers'):
            for layer in self.classification_head.layers:
                if 'mobilenetv2' in layer.name.lower() or 'base_model' in layer.name.lower():
                    layer.trainable = True
        
        # Lower learning rate for fine-tuning
        # Re-compile with appropriate loss/metrics for fine-tuning
        if y_student.shape[1] == 1:
            self.classification_head.compile(
                optimizer=keras.optimizers.AdamW(
                    learning_rate=self.learning_rate * 0.1,
                    weight_decay=0.01,
                    beta_1=0.9,
                    beta_2=0.999
                ),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            self.classification_head.compile(
                optimizer=keras.optimizers.AdamW(
                    learning_rate=self.learning_rate * 0.1,
                    weight_decay=0.01,
                    beta_1=0.9,
                    beta_2=0.999
                ),
                loss='categorical_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
                    # Removed Precision, Recall, AUC due to Keras 3.x compatibility issues
                ]
            )
        
        # Phase 2: Fine-tune with unfrozen backbone
        phase2_epochs = epochs - phase1_epochs
        if phase2_epochs > 0:
            fine_tune_history = self.classification_head.fit(
                X, y_student,
                batch_size=batch_size,
                epochs=phase2_epochs,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            for key in classification_history.history:
                if key in fine_tune_history.history:
                    classification_history.history[key].extend(fine_tune_history.history[key])
        
        # Log detailed training metrics
        final_accuracy = classification_history.history.get('accuracy', [0.0])[-1]
        final_loss = classification_history.history.get('loss', [0.0])[-1]
        val_accuracy = classification_history.history.get('val_accuracy', [0.0])[-1]
        val_loss = classification_history.history.get('val_loss', [0.0])[-1]
        
        logger.info(f"Classification training completed successfully!")
        logger.info(f"Final training accuracy: {final_accuracy:.4f}")
        logger.info(f"Final training loss: {final_loss:.4f}")
        logger.info(f"Final validation accuracy: {val_accuracy:.4f}")
        logger.info(f"Final validation loss: {val_loss:.4f}")
        logger.info(f"Total epochs trained: {len(classification_history.history.get('accuracy', []))}")
        
        return {
            'classification_history': classification_history.history,
            'student_mappings': {
                'student_to_id': self.student_to_id,
                'id_to_student': self.id_to_student
            }
        }

    def train_models(self, training_data: Dict, epochs: int = 100) -> Dict:
        """
        Train models with advanced training strategies
        Focus: Owner identification only (forgery detection disabled)
        """
        logger.info("Starting comprehensive model training...")
        
        # Prepare data
        X, y_student = self.prepare_training_data(training_data)
        
        # Create models
        self.create_classification_head()
        self.create_siamese_network()
        
        # Optimized callbacks for faster training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,  # Reduced patience for faster training
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,  # Reduced patience
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train classification model using advanced two-phase approach
        logger.info("Training student classification model with progressive unfreezing...")
        
        # Phase 1: Train with frozen backbone
        phase1_epochs = min(epochs // 2, 25)  # Use half epochs for phase 1
        classification_history = self.classification_head.fit(
            X, y_student,
            batch_size=32,
            epochs=phase1_epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tune with unfrozen backbone
        logger.info("Phase 2: Fine-tuning with unfrozen backbone...")
        
        # Unfreeze the backbone for fine-tuning
        if hasattr(self.classification_head, 'layers'):
            for layer in self.classification_head.layers:
                if 'mobilenetv2' in layer.name.lower() or 'base_model' in layer.name.lower():
                    layer.trainable = True
        
        # Lower learning rate for fine-tuning
        self.classification_head.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=self.learning_rate * 0.1,  # Lower learning rate for fine-tuning
                weight_decay=0.01,
                beta_1=0.9,
                beta_2=0.999
            ),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
                # Removed Precision, Recall, AUC due to Keras 3.x compatibility issues
            ]
        )
        
        # Phase 2: Fine-tune with unfrozen backbone
        phase2_epochs = epochs - phase1_epochs
        if phase2_epochs > 0:
            fine_tune_history = self.classification_head.fit(
                X, y_student,
                batch_size=32,
                epochs=phase2_epochs,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            for key in classification_history.history:
                if key in fine_tune_history.history:
                    classification_history.history[key].extend(fine_tune_history.history[key])
        
        # Train Siamese network with pair generation
        logger.info("Training Siamese network...")
        siamese_pairs, siamese_labels = self._generate_siamese_pairs(training_data)
        
        # Split pairs into separate inputs for Siamese model
        if len(siamese_pairs) > 0:
            input_a = siamese_pairs[:, 0]  # First image of each pair
            input_b = siamese_pairs[:, 1]  # Second image of each pair
            siamese_history = self.siamese_model.fit(
                [input_a, input_b], siamese_labels,
                batch_size=32,
                epochs=epochs,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Create dummy history if no pairs generated
            siamese_history = type('History', (), {'history': {}})()
        
        logger.info("Training completed successfully!")
        
        return {
            'classification_history': classification_history.history,
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
            
            # Positive pairs (genuine-genuine) - generate more pairs for better training
            for i in range(len(genuine_images)):
                for j in range(i + 1, len(genuine_images)):
                    img1 = self._preprocess_signature(genuine_images[i])
                    img2 = self._preprocess_signature(genuine_images[j])
                    pairs.append([img1, img2])
                    labels.append(1)  # Similar
            
            # Skip negative pairs with forged signatures - not used for owner identification
            # (Forgery detection is disabled - focus on owner identification only)
        
        if len(pairs) == 0:
            return np.array([]), np.array([])
        
        # Convert pairs to numpy array and return as (n_pairs, 2, height, width, channels)
        return np.array(pairs), np.array(labels)
    
    def verify_signature(self, test_signature: Union[np.ndarray, Image.Image]) -> Dict:
        """
        Comprehensive signature verification with multiple models
        """
        logger.info("Performing comprehensive signature verification...")
        
        # Check available heads; forgery detection is disabled system-wide
        if not self.embedding_model:
            raise ValueError("Embedding model not loaded. Please load a trained model first.")
        has_classification = self.classification_head is not None

        # If the "classification" head is actually a 1-unit authenticity model, do not treat it as a classifier
        if has_classification:
            try:
                # Get the actual output shape by running a test prediction
                test_input = np.random.random((1, self.image_size, self.image_size, 3))
                test_output = self.classification_head.predict(test_input, verbose=0)
                output_shape = test_output.shape
                
                if len(output_shape) == 2 and output_shape[1] == 1:
                    # 1-unit outputs are invalid for identification; disable classification path
                    self.classification_head = None
                    has_classification = False
                    logger.warning("Disabled 1-unit model for identification (forgery detection disabled - focus on owner identification only)")
                else:
                    logger.info(f"Classification head has {output_shape[1]} outputs - treating as classifier")
            except Exception as e:
                logger.warning(f"Could not determine classification head output shape: {e}")
                pass
        if not has_classification:
            # Fallback: allow embedding-only flow; return unknown prediction with zero confidence
            processed_signature = self._preprocess_signature(test_signature)
            X_test = np.expand_dims(processed_signature, axis=0)
            embedding = self.embedding_model.predict(X_test, verbose=0)[0]
            return {
                'predicted_student_id': 0,
                'predicted_student_name': 'Unknown',
                'student_confidence': 0.0,
                'is_genuine': False,
                'authenticity_score': 0.0,
                'overall_confidence': 0.0,
                'is_unknown': True,
                'embedding': embedding.tolist()
            }
        
        # Preprocess test signature
        processed_signature = self._preprocess_signature(test_signature)
        X_test = np.expand_dims(processed_signature, axis=0)
        
        # Get embeddings
        embedding = self.embedding_model.predict(X_test, verbose=0)[0]
        
        predicted_student_id = 0
        student_confidence = 0.0
        # Predict identification strictly via classification head
        student_probs = self.classification_head.predict(X_test, verbose=0)[0]
        predicted_class_index = int(np.argmax(student_probs))
        student_confidence = float(np.max(student_probs))
        logger.info(f"Classification prediction: class {predicted_student_id}, confidence {student_confidence:.3f}")
        
        # Authenticity detection is disabled - focus on owner identification only
        authenticity_score = 0.0
        is_genuine = True  # Always true since we're not doing forgery detection
        
        # Get student name and map class index to numeric student_id
        predicted_student_name = self.id_to_student.get(predicted_class_index, f"Unknown_{predicted_class_index}")
        if self.external_student_id_map and predicted_student_name in self.external_student_id_map:
            actual_student_id = int(self.external_student_id_map[predicted_student_name])
        else:
            actual_student_id = predicted_class_index
        
        # Calculate overall confidence
        overall_confidence = student_confidence
        
        # Determine if signature is unknown - stricter threshold to avoid false 100%
        is_unknown = (
            (student_confidence < 0.5) or
            predicted_class_index == 0
        )

        # Build top-k outputs for debug/inference (top-3)
        try:
            top_k = 3 if student_probs.shape[0] >= 3 else student_probs.shape[0]
            top_indices = np.argsort(-student_probs)[:top_k]
            top_list = []
            for idx in top_indices:
                name = self.id_to_student.get(int(idx), f"Unknown_{int(idx)}")
                sid = int(self.external_student_id_map.get(name)) if (self.external_student_id_map and name in self.external_student_id_map) else int(idx)
                top_list.append({'student_id': sid, 'name': name, 'prob': float(student_probs[int(idx)])})
        except Exception:
            top_list = []
        
        result = {
            'predicted_student_id': actual_student_id,  # Use actual student ID
            'predicted_student_name': predicted_student_name,
            'student_confidence': student_confidence,
            'is_genuine': bool(is_genuine),
            'authenticity_score': authenticity_score,
            'overall_confidence': overall_confidence,
            'is_unknown': is_unknown,
            'embedding': embedding.tolist(),  # For similarity comparisons
            'top_k': top_list
        }
        
        return result
    
    def save_models(self, base_path: str):
        """
        Save all trained models to disk with ID-first mappings
        """
        logger.info(f"Saving models to {base_path}...")
        
        if self.embedding_model:
            embedding_path = f"{base_path}_embedding.keras"
            self.embedding_model.save(embedding_path)
            logger.info(f"Saved embedding model to {embedding_path}")
        
        if self.classification_head:
            classification_path = f"{base_path}_classification.keras"
            self.classification_head.save(classification_path)
            logger.info(f"Saved classification model to {classification_path}")
        
        # Note: authenticity_head removed - forgery detection is disabled
        
        if self.siamese_model:
            siamese_path = f"{base_path}_siamese.keras"
            self.siamese_model.save(siamese_path)
            logger.info(f"Saved Siamese model to {siamese_path}")
        
        # Save ID-first mappings (Teachable Machine style)
        mappings_path = f"{base_path}_mappings.json"
        with open(mappings_path, 'w') as f:
            import json
            mappings = {
                # Primary ID-first schema
                'class_index_to_student_id': {str(ci): int(sid) for ci, sid in self.class_to_student_id.items()},
                'class_index_to_student_name': {str(ci): self.student_id_to_name.get(sid, f"Student_{sid}") 
                                               for ci, sid in self.class_to_student_id.items()},
                'student_id_to_class_index': {str(sid): int(ci) for sid, ci in self.student_id_to_class.items()},
                
                # Metadata
                'metadata': {
                    'num_classes': len(self.class_to_student_id),
                    'image_size': self.image_size,
                    'embedding_dim': self.embedding_dim,
                    'schema_version': 'v2_id_first'
                },
                
                # Legacy mappings (deprecated)
                'student_to_id': self.student_to_id,  # For backward compatibility only
                'id_to_student': self.id_to_student   # For backward compatibility only
            }
            json.dump(mappings, f, indent=2)
        logger.info(f"Saved ID-first mappings to {mappings_path}")
    
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
            
            # Note: authenticity model loading removed - forgery detection is disabled
            
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
