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

class SignatureVerificationModel:
    """
    AI system for Signature Verification (student recognition + authenticity)
    """
    
    def __init__(self, max_students: int = 150):
        self.max_students = max_students
        self.student_model = None
        self.authenticity_model = None
        self.student_to_id = {}
        self.id_to_student = {}
        self.image_size = settings.MODEL_IMAGE_SIZE
        self.batch_size = settings.MODEL_BATCH_SIZE
        self.epochs = settings.MODEL_EPOCHS
        self.learning_rate = settings.MODEL_LEARNING_RATE
        
        logger.info(f"Initialized signature AI system for up to {max_students} students")
    
    def create_student_recognition_model(self):
        """Create the model for recognizing individual students from signatures"""
        logger.info("Building student recognition model...")
        
        input_layer = layers.Input(shape=(self.image_size, self.image_size, 3), name='signature_input')
        
        backbone = keras.applications.MobileNetV2(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        for layer in backbone.layers[:-40]:
            layer.trainable = False
        
        preprocessed = keras.applications.mobilenet_v2.preprocess_input(input_layer)
        features = backbone(preprocessed, training=False)
        
        attention = layers.Conv2D(1, (1, 1), activation='sigmoid', name='spatial_attention')(features)
        attended_features = layers.Multiply(name='attention_application')([features, attention])
        
        pooled_features = layers.GlobalAveragePooling2D(name='global_pooling')(attended_features)
        
        x = layers.Dense(1024, activation='relu', name='fc1')(pooled_features)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        
        x = layers.Dense(512, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.Dropout(0.3, name='dropout2')(x)
        
        x = layers.Dense(256, activation='relu', name='fc3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.Dropout(0.2, name='dropout3')(x)
        
        student_output = layers.Dense(
            self.max_students,
            activation='softmax',
            name='student_classification'
        )(x)
        
        model = keras.Model(inputs=input_layer, outputs=student_output, name='signature_student_recognition')
        
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            weight_decay=1e-4
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        logger.info(f"Student recognition model created; params: {model.count_params():,}")
        return model
    
    def create_authenticity_detection_model(self):
        """Create a model to detect if a signature is genuine or forged"""
        logger.info("Building authenticity detection model...")
        
        input_layer = layers.Input(shape=(self.image_size, self.image_size, 3), name='signature_input')
        
        backbone = keras.applications.MobileNetV2(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        
        preprocessed = keras.applications.mobilenet_v2.preprocess_input(input_layer)
        features = backbone(preprocessed, training=False)
        
        attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(features)
        attended_features = layers.Multiply()([features, attention])
        
        pooled_features = layers.GlobalAveragePooling2D()(attended_features)
        
        x = layers.Dense(512, activation='relu')(pooled_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        authenticity_output = layers.Dense(1, activation='sigmoid', name='authenticity_classification')(x)
        
        model = keras.Model(inputs=input_layer, outputs=authenticity_output, name='signature_authenticity_detection')
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, weight_decay=1e-4)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        logger.info(f"Authenticity model created; params: {model.count_params():,}")
        return model
    
    def prepare_training_data(self, training_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for the system"""
        logger.info("Preparing training data...")
        
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
            
            for img in signatures['genuine']:
                processed_img = self.preprocess_image(img)
                all_images.append(processed_img)
                student_labels.append(student_id)
                authenticity_labels.append(1)
            
            for img in signatures['forged']:
                processed_img = self.preprocess_image(img)
                all_images.append(processed_img)
                student_labels.append(student_id)
                authenticity_labels.append(0)
        
        X = np.array(all_images, dtype=np.float32)
        y_student = np.array(student_labels, dtype=np.int32)
        y_authenticity = np.array(authenticity_labels, dtype=np.float32)
        
        y_student_onehot = tf.keras.utils.to_categorical(y_student, num_classes=self.max_students)
        
        logger.info(f"Prepared {len(all_images)} images across {len(self.student_to_id)} students")
        return X, y_student_onehot, y_authenticity
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Preprocess image to model input format"""
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = tf.cast(image, tf.float32)
        if tf.reduce_max(image) <= 1.0:
            image = image * 255.0
        return image.numpy()
    
    def train_system(self, training_data: Dict) -> Dict:
        """Train both student recognition and authenticity models"""
        logger.info("Starting training...")
        X, y_student, y_authenticity = self.prepare_training_data(training_data)
        
        self.student_model = self.create_student_recognition_model()
        self.authenticity_model = self.create_authenticity_detection_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_student_model.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        student_history = self.student_model.fit(
            X, y_student,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        authenticity_history = self.authenticity_model.fit(
            X, y_authenticity,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        return {
            'student_history': student_history.history,
            'authenticity_history': authenticity_history.history,
            'student_model': self.student_model,
            'authenticity_model': self.authenticity_model
        }
    
    def verify_signature(self, test_signature: Union[np.ndarray, Image.Image]) -> Dict:
        """Identify student and determine authenticity with confidences"""
        logger.info("Verifying signature...")
        processed_signature = self.preprocess_image(test_signature)
        X_test = np.expand_dims(processed_signature, axis=0)
        
        student_probs = self.student_model.predict(X_test, verbose=0)[0]
        predicted_student_id = int(np.argmax(student_probs))
        student_confidence = float(np.max(student_probs))
        
        authenticity_score = float(self.authenticity_model.predict(X_test, verbose=0)[0][0])
        is_genuine = authenticity_score > 0.5
        
        predicted_student_name = self.id_to_student.get(predicted_student_id, f"Unknown_{predicted_student_id}")
        overall_confidence = (student_confidence + authenticity_score) / 2.0
        
        result = {
            'predicted_student_id': predicted_student_id,
            'predicted_student_name': predicted_student_name,
            'student_confidence': student_confidence,
            'is_genuine': bool(is_genuine),
            'authenticity_score': authenticity_score,
            'overall_confidence': overall_confidence,
            'is_unknown': student_confidence < 0.3 or overall_confidence < 0.4
        }
        return result
    
    def save_models(self, base_path: str):
        """Save trained models and mappings"""
        if self.student_model:
            student_path = f"{base_path}_student_model.keras"
            self.student_model.save(student_path)
            logger.info(f"Student model saved to {student_path}")
        if self.authenticity_model:
            authenticity_path = f"{base_path}_authenticity_model.keras"
            self.authenticity_model.save(authenticity_path)
            logger.info(f"Authenticity model saved to {authenticity_path}")
        import json
        mappings_path = f"{base_path}_student_mappings.json"
        with open(mappings_path, 'w') as f:
            json.dump({
                'student_to_id': self.student_to_id,
                'id_to_student': self.id_to_student
            }, f, indent=2)
        logger.info(f"Student mappings saved to {mappings_path}")
    
    def load_models(self, base_path: str):
        """Load trained models and mappings if present"""
        try:
            student_path = f"{base_path}_student_model.keras"
            if os.path.exists(student_path):
                self.student_model = keras.models.load_model(student_path)
                logger.info(f"Student model loaded from {student_path}")
            authenticity_path = f"{base_path}_authenticity_model.keras"
            if os.path.exists(authenticity_path):
                self.authenticity_model = keras.models.load_model(authenticity_path)
                logger.info(f"Authenticity model loaded from {authenticity_path}")
            mappings_path = f"{base_path}_student_mappings.json"
            if os.path.exists(mappings_path):
                import json
                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                self.student_to_id = mappings['student_to_id']
                self.id_to_student = {int(k): v for k, v in mappings['id_to_student'].items()}
                logger.info(f"Student mappings loaded from {mappings_path}")
            logger.info("Signature models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load signature models: {e}")
            return False
