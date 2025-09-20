#!/usr/bin/env python3
import sys
import os
import json
import boto3
import numpy as np
from PIL import Image
import io
import traceback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tempfile
import shutil
import base64
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 60)
print("STARTING TRAINING SCRIPT")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Arguments: {sys.argv}")

if len(sys.argv) != 4:
    print("Usage: train_gpu.py <training_data_key> <job_id> <student_id>")
    sys.exit(1)

training_data_key = sys.argv[1]
job_id = sys.argv[2]
student_id = int(sys.argv[3])

print(f"Training data key: {training_data_key}")
print(f"Job ID: {job_id}")
print(f"Student ID: {student_id}")

# GPU Configuration with memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Found {len(gpus)} GPU(s), using GPU acceleration")
    except RuntimeError as e:
        logger.error(f"GPU setup error: {e}")
else:
    logger.warning("No GPU found, using CPU")

class SignaturePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.processed_count = 0
        self.error_count = 0
    def preprocess_signature(self, img_data, debug_name="unknown"):
        try:
            img = None
            if isinstance(img_data, dict):
                # New wrapped format: {array: [...], shape: [...]} or {base64: "..."}
                if 'base64' in img_data:
                    try:
                        b = base64.b64decode(img_data['base64'])
                        img = Image.open(io.BytesIO(b))
                        print("  Loaded base64 image for {}".format(debug_name))
                    except Exception as e:
                        print("  Failed to decode base64 for {}: {}".format(debug_name, e))
                        return None
                elif 'array' in img_data:
                    arr = np.array(img_data['array'], dtype=np.float32)
                    shape = img_data.get('shape') or []
                    # Try to reshape using provided shape
                    try:
                        if shape:
                            arr = arr.reshape(shape)
                        # If still flat, try to square or 3-channel heuristics
                        if arr.ndim == 1:
                            total = arr.size
                            side = int(np.sqrt(total))
                            if side*side == total:
                                arr = arr.reshape(side, side)
                            elif (total % 3) == 0:
                                side = int(np.sqrt(total//3))
                                if side*side*3 == total:
                                    arr = arr.reshape(side, side, 3)
                        if arr.ndim == 2:
                            arr = np.stack([arr]*3, axis=-1)
                        elif arr.ndim == 3 and arr.shape[2] == 1:
                            arr = np.repeat(arr, 3, axis=2)
                        img = Image.fromarray(np.clip(arr*255 if arr.max()<=1.0 else arr, 0, 255).astype(np.uint8))
                        print("  Reconstructed image from wrapped array for {} with shape {}".format(debug_name, arr.shape))
                    except Exception as e:
                        print("  Failed to reconstruct image for {}: {}".format(debug_name, e))
                        return None
                else:
                    print("  Unknown wrapped dict for {}".format(debug_name))
                    return None
            elif isinstance(img_data, str):
                try:
                    if img_data.startswith('data:'):
                        img_data = img_data.split(',')[1]
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    print("  Successfully loaded base64 image for {}".format(debug_name))
                except Exception as e:
                    print("  Failed to decode base64 for {}: {}".format(debug_name, e))
                    return None
            elif isinstance(img_data, list):
                img_array = np.array(img_data, dtype=np.float32)
                print("  Processing array data for {}, shape: {}".format(debug_name, img_array.shape))
                if len(img_array.shape) == 1:
                    total_pixels = len(img_array)
                    side = int(np.sqrt(total_pixels))
                    if side * side == total_pixels:
                        img_array = img_array.reshape(side, side)
                        print("    Reshaped flat array to {}".format(img_array.shape))
                    else:
                        common_sizes = [(224,224),(256,256),(128,128),(64,64)]
                        reshaped = False
                        for h,w in common_sizes:
                            if h*w == total_pixels:
                                img_array = img_array.reshape(h,w)
                                print("    Reshaped to common size: {}".format(img_array.shape))
                                reshaped = True
                                break
                        if not reshaped:
                            for h,w in common_sizes:
                                if h*w*3 == total_pixels:
                                    img_array = img_array.reshape(h,w,3)
                                    print("    Reshaped to 3-channel: {}".format(img_array.shape))
                                    reshaped = True
                                    break
                        if not reshaped:
                            print("    Cannot reshape array of size {} to known image format".format(total_pixels))
                            return None
                elif len(img_array.shape) == 3:
                    if img_array.shape[0] in [1,3,4]:
                        img_array = np.transpose(img_array, (1,2,0))
                        print("    Transposed from CHW to HWC: {}".format(img_array.shape))
                    elif img_array.shape[2] not in [1,3,4]:
                        print("    Unexpected shape: {}".format(img_array.shape))
                        return None
                if img_array.max() <= 1.0 and img_array.min() >= 0.0:
                    img_array = (img_array * 255).astype(np.uint8)
                    print("    Scaled normalized values to 0-255 range")
                elif img_array.max() > 255 or img_array.min() < 0:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    print("    Clamped values to 0-255 range")
                else:
                    img_array = img_array.astype(np.uint8)
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                    print("    Converted grayscale to RGB: {}".format(img_array.shape))
                elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                    img_array = np.repeat(img_array, 3, axis=2)
                    print("    Converted single channel to RGB: {}".format(img_array.shape))
                try:
                    img = Image.fromarray(img_array)
                    print("    Created PIL image from array: {}".format(img.size))
                except Exception as e:
                    print("    Failed to create PIL image: {}".format(e))
                    return None
            elif hasattr(img_data, 'size'):
                img = img_data
                print("  Using existing PIL image for {}: {}".format(debug_name, img.size))
            else:
                print("  Unknown image data type for {}: {}".format(debug_name, type(img_data)))
                return None
            if img is None:
                return None
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("    Converted to RGB mode")
            original_size = img.size
            # Pillow compatibility: handle older versions without Image.Resampling
            try:
                resample_filter = Image.Resampling.LANCZOS
            except Exception:
                resample_filter = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BICUBIC
            img = img.resize(self.target_size, resample_filter)
            print("    Resized from {} to {}".format(original_size, img.size))
            img_array = np.array(img, dtype=np.float32) / 255.0
            self.processed_count += 1
            return img_array
        except Exception as e:
            print("  Error processing {}: {}".format(debug_name, str(e)))
            print("  Stack trace: {}".format(traceback.format_exc()))
            self.error_count += 1
            return None

class SignatureEmbeddingModel:
    def __init__(self, max_students=150, image_size=224):
        self.max_students = max_students
        self.embedding_dim = 128
        self.image_size = image_size
        self.student_to_id = {}  # Maps student_id -> class_index
        self.id_to_student = {}  # Maps class_index -> student_id
        self.id_to_name = {}     # Maps student_id -> student_name
        self.embedding_model = None
        self.classification_head = None
        self.siamese_model = None
        self.base_model = None  # For transfer learning
    def create_augmentation_pipeline(self):
        """Create tf.data augmentation pipeline for training"""
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),  # Small rotation for signatures
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
        ], name='augmentation')
    
    def create_tf_data_pipeline(self, X, y, batch_size=32, is_training=True):
        """Create efficient tf.data pipeline with augmentation"""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(X))
            augmentation = self.create_augmentation_pipeline()
            dataset = dataset.map(
                lambda x, y: (augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def create_model_with_transfer_learning(self, num_classes):
        """Create model with MobileNetV2 transfer learning"""
        # Input layer
        inputs = keras.Input(shape=(self.image_size, self.image_size, 3))
        
        # Use MobileNetV2 as base model (lightweight and effective)
        self.base_model = keras.applications.MobileNetV2(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Initially freeze the base model
        self.base_model.trainable = False
        
        # Preprocessing for MobileNetV2
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Pass through base model
        x = self.base_model(x, training=False)
        
        # Custom head for signature classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def train_models(self, training_data, epochs=50, validation_split=0.2):
        logger.info("Starting Teachable Machine-style training...")
        all_images = []
        all_labels = []
        all_student_ids = []
        
        # Process training data with ID-first approach
        logger.info(f"Processing {len(training_data)} students...")
        class_idx = 0
        for student_key, data in training_data.items():
            # Extract student ID from key (format: "StudentID_Name" or just "StudentID")
            if '_' in student_key:
                student_id = int(student_key.split('_')[0])
                student_name = '_'.join(student_key.split('_')[1:])
            else:
                try:
                    student_id = int(student_key)
                    student_name = f"Student_{student_id}"
                except:
                    # Fallback: use sequential ID
                    student_id = class_idx + 1000
                    student_name = student_key
            
            # Store ID-first mappings
            self.student_to_id[student_id] = class_idx
            self.id_to_student[class_idx] = student_id
            self.id_to_name[student_id] = student_name
            
            genuine_count = len(data.get('genuine', []))
            forged_count = len(data.get('forged', []))
            logger.info(f"Student ID {student_id} ('{student_name}', class {class_idx}): {genuine_count} genuine, {forged_count} forged")
            
            # Process genuine images
            for i, img in enumerate(data.get('genuine', [])):
                if img is not None:
                    all_images.append(img)
                    all_labels.append(class_idx)
                    all_student_ids.append(student_id)
            
            # Process forged images (if any)
            for i, img in enumerate(data.get('forged', [])):
                if img is not None:
                    all_images.append(img)
                    all_labels.append(class_idx)
                    all_student_ids.append(student_id)
            
            class_idx += 1
        logger.info(f"Total images for training: {len(all_images)}")
        logger.info(f"Unique classes: {len(set(all_labels))}")
        
        if len(all_images) == 0:
            raise ValueError("No valid training samples found after processing")
        
        # Handle small datasets with augmentation
        if len(all_images) < 10:
            logger.warning(f"Small dataset ({len(all_images)} samples). Applying heavy augmentation...")
            validation_split = 0.1  # Still use validation for small datasets
        
        logger.info("Converting to numpy arrays...")
        X = np.array(all_images, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        
        # Stratified train/validation split
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, stratify=y, random_state=42
            )
            logger.info(f"Train set: {X_train.shape}, Val set: {X_val.shape}")
        else:
            X_train, X_val = X, X[:1]  # Use first sample as dummy validation
            y_train, y_val = y, y[:1]
        
        # Convert labels to categorical
        num_classes = len(np.unique(y))
        y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
        y_val_cat = keras.utils.to_categorical(y_val, num_classes=num_classes)
        
        # Compute class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        logger.info(f"Class weights: {class_weight_dict}")
        # Create model with transfer learning
        logger.info("Creating model with MobileNetV2 transfer learning...")
        model = self.create_model_with_transfer_learning(num_classes)
        
        # Store as classification head
        self.classification_head = model
        
        # Extract embedding model (without final classification layer)
        self.embedding_model = keras.Model(
            inputs=model.input,
            outputs=model.layers[-4].output  # Before the final Dense layer
        )
        
        logger.info("Model architecture:")
        model.summary(logging=True)
        # Create tf.data pipelines
        batch_size = min(32, max(2, len(X_train) // 4))  # Adaptive batch size
        train_dataset = self.create_tf_data_pipeline(X_train, y_train_cat, batch_size, is_training=True)
        val_dataset = self.create_tf_data_pipeline(X_val, y_val_cat, batch_size, is_training=False)
        
        # Two-phase training strategy
        logger.info("\n=== PHASE 1: Training head with frozen backbone ===")
        
        # Phase 1: Train only the head (frozen backbone)
        self.base_model.trainable = False
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=min(3, num_classes), name='top_k_accuracy')]
        )
        
        # Callbacks for Phase 1
        phase1_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                '/tmp/best_phase1.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train Phase 1
        phase1_epochs = min(epochs // 2, 20)
        history_phase1 = model.fit(
            train_dataset,
            epochs=phase1_epochs,
            validation_data=val_dataset,
            callbacks=phase1_callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        logger.info("\n=== PHASE 2: Fine-tuning with unfrozen backbone ===")
        
        # Phase 2: Fine-tune the last layers of backbone
        self.base_model.trainable = True
        # Freeze all layers except the last 20
        for layer in self.base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=min(3, num_classes), name='top_k_accuracy')]
        )
        
        # Callbacks for Phase 2
        phase2_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                '/tmp/best_phase2.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train Phase 2
        phase2_epochs = epochs - phase1_epochs
        history_phase2 = model.fit(
            train_dataset,
            epochs=phase2_epochs,
            validation_data=val_dataset,
            callbacks=phase2_callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Combine histories
        history = {}
        for key in history_phase1.history:
            history[key] = history_phase1.history[key] + history_phase2.history[key]
        
        # Print final training results
        final_loss = history['loss'][-1]
        final_accuracy = history['accuracy'][-1]
        final_val_loss = history.get('val_loss', [0])[-1]
        final_val_accuracy = history.get('val_accuracy', [0])[-1]
        epochs_trained = len(history['loss'])
        
        logger.info(f"\n=== TRAINING COMPLETED ===")
        logger.info(f"Total epochs trained: {epochs_trained}")
        logger.info(f"Final metrics - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
        logger.info(f"Validation metrics - Val Loss: {final_val_loss:.4f}, Val Accuracy: {final_val_accuracy:.4f}")
        
        # Load best model if checkpoint exists
        if os.path.exists('/tmp/best_phase2.keras'):
            logger.info("Loading best model from checkpoint...")
            self.classification_head = keras.models.load_model('/tmp/best_phase2.keras')
            # Re-extract embedding model
            self.embedding_model = keras.Model(
                inputs=self.classification_head.input,
                outputs=self.classification_head.layers[-4].output
            )
        
        return {
            'classification_history': history,
            'siamese_history': {},
            'training_metadata': {
                'num_classes': num_classes,
                'total_samples': len(all_images),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'class_weights': class_weight_dict,
                'phase1_epochs': phase1_epochs,
                'phase2_epochs': phase2_epochs
            }
        }

    def save_models(self, base_path):
        try:
            if self.embedding_model is not None:
                emb_path = f"{base_path}_embedding.keras"
                self.embedding_model.save(emb_path)
                logger.info(f"Saved embedding model to {emb_path}")
            
            if self.classification_head is not None:
                cls_path = f"{base_path}_classification.keras"
                self.classification_head.save(cls_path)
                logger.info(f"Saved classification model to {cls_path}")
            
            # Save ID-first mappings (Teachable Machine style)
            map_path = f"{base_path}_mappings.json"
            mappings = {
                'class_index_to_student_id': {str(ci): int(sid) for ci, sid in self.id_to_student.items()},
                'class_index_to_student_name': {str(ci): self.id_to_name.get(sid, f"Student_{sid}") for ci, sid in self.id_to_student.items()},
                'student_id_to_class_index': {str(sid): int(ci) for sid, ci in self.student_to_id.items()},
                'metadata': {
                    'num_classes': len(self.id_to_student),
                    'image_size': self.image_size,
                    'embedding_dim': self.embedding_dim
                }
            }
            with open(map_path, 'w') as f:
                json.dump(mappings, f, indent=2)
            logger.info(f"Saved ID-first mappings to {map_path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            traceback.print_exc()
            raise

def train_on_gpu(training_data_key, job_id, student_id):
    try:
        print("Starting training for job {}".format(job_id))
        print("Training data key: {}".format(training_data_key))
        with open(training_data_key, 'r') as f:
            training_data_raw = json.load(f)
        logger.info("Raw training data contains {} students".format(len(training_data_raw)))
        preprocessor = SignaturePreprocessor(target_size=(224, 224))
        model_manager = SignatureEmbeddingModel(max_students=150, image_size=224)
        logger.info("Processing training data with ID-first approach...")
        processed_data = {}
        
        # Process each student
        class_idx = 0
        logger.info(f"Processing {len(training_data_raw)} students from training data")
        
        for student_key, data in training_data_raw.items():
            logger.info(f"Processing student key: {student_key}")
            
            # Skip empty data
            if not data or (not data.get('genuine') and not data.get('forged')):
                logger.warning(f"Skipping student {student_key} - no images found")
                continue
            
            # Parse student ID and name from key
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
            
            # Store mappings
            model_manager.student_id_to_class[student_id] = class_idx
            model_manager.class_to_student_id[class_idx] = student_id
            model_manager.student_id_to_name[student_id] = student_name
            
            # Legacy mappings for compatibility
            model_manager.student_to_id[student_name] = class_idx
            model_manager.id_to_student[class_idx] = student_name
            
            genuine_count = len(data.get('genuine', []))
            forged_count = len(data.get('forged', []))
            logger.info(f"Student ID {student_id} ('{student_name}', class {class_idx}): {genuine_count} genuine, {forged_count} forged")
            
            # Use ID_Name format for processed data key
            processed_key = f"{student_id}_{student_name}"
            genuine_images = []
            forged_images = []
            # Accept both new and legacy keys
            genuine_raw = data.get('genuine')
            if not genuine_raw:
                genuine_raw = data.get('genuine_images', [])
            print("  Found {} genuine images".format(len(genuine_raw)))
            for i, img_data in enumerate(genuine_raw):
                print("    Processing genuine image {}/{}".format(i+1, len(genuine_raw)))
                processed_img = preprocessor.preprocess_signature(img_data, debug_name="{}_genuine_{}".format(student_name, i))
                if processed_img is not None:
                    genuine_images.append(processed_img)
                    print("      Successfully processed")
                    print("      \u2713 Successfully processed")
                else:
                    print("      x Failed to process")
            forged_raw = data.get('forged')
            if not forged_raw:
                forged_raw = data.get('forged_images', [])
            print("  Found {} forged images".format(len(forged_raw)))
            for i, img_data in enumerate(forged_raw):
                print("    Processing forged image {}/{}".format(i+1, len(forged_raw)))
                processed_img = preprocessor.preprocess_signature(img_data, debug_name="{}_forged_{}".format(student_name, i))
                if processed_img is not None:
                    forged_images.append(processed_img)
                    print("      \u2713 Successfully processed")
                else:
                    print("      x Failed to process")
            processed_data[processed_key] = {'genuine': genuine_images, 'forged': forged_images}
            total_processed = len(genuine_images) + len(forged_images)
            total_raw = len(genuine_raw) + len(forged_raw)
            success_rate = (total_processed / total_raw * 100) if total_raw > 0 else 0
            print("  Final: {} genuine, {} forged".format(len(genuine_images), len(forged_images)))
            print("  Success rate: {}/{} ({:.1f}%)".format(total_processed, total_raw, success_rate))
            
            # Increment class index for next student
            class_idx += 1
        logger.info("\n=== PREPROCESSING SUMMARY ===")
        logger.info("Total images processed successfully: {}".format(preprocessor.processed_count))
        logger.info("Total processing errors: {}".format(preprocessor.error_count))
        total_samples = sum(len(d['genuine']) + len(d['forged']) for d in processed_data.values())
        logger.info("Total processed samples available for training: {}".format(total_samples))
        if total_samples == 0:
            raise ValueError("No valid training samples found after processing")
        validation_split = 0.0 if total_samples < 5 else 0.2
        logger.info("\n=== STARTING MODEL TRAINING ===")
        training_result = model_manager.train_models(processed_data, epochs=50, validation_split=validation_split)
        logger.info("Training completed! Saving models and artifacts...")
        temp_dir = '/tmp/{}_models'.format(job_id)
        os.makedirs(temp_dir, exist_ok=True)
        model_manager.save_models('{}/signature_model'.format(temp_dir))
        
        # Upload all model files
        model_files = ['embedding','classification']
        model_urls = {}
        for model_type in model_files:
            file_path = '{}/signature_model_{}.keras'.format(temp_dir, model_type)
            if os.path.exists(file_path):
                s3_key = 'models/{}/{}.keras'.format(job_id, model_type)
                s3.upload_file(file_path, bucket, s3_key)
                model_urls[model_type] = 'https://{}.s3.amazonaws.com/{}'.format(bucket, s3_key)
                logger.info("Uploaded {} model to S3: {}".format(model_type, s3_key))
            else:
                logger.warning("WARNING: {} model file not found: {}".format(model_type, file_path))
        
        # Upload ID-first mappings
        mappings_path = '{}/signature_model_mappings.json'.format(temp_dir)
        if os.path.exists(mappings_path):
            s3_key = 'models/{}/mappings.json'.format(job_id)
            s3.upload_file(mappings_path, bucket, s3_key)
            model_urls['mappings'] = 'https://{}.s3.amazonaws.com/{}'.format(bucket, s3_key)
            logger.info("Uploaded ID-first mappings to S3: {}".format(s3_key))
        # Save SavedModel format for production deployment
        try:
            savedmodel_dir = '{}/savedmodel'.format(temp_dir)
            model_manager.classification_head.save(savedmodel_dir, save_format='tf')
            logger.info("Saved SavedModel to {}".format(savedmodel_dir))
            
            # Zip the SavedModel directory
            import zipfile
            zip_path = '{}/savedmodel.zip'.format(temp_dir)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(savedmodel_dir):
                    for f in files:
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, start=savedmodel_dir)
                        zf.write(full, arcname=rel)
            
            s3_key = 'models/{}/savedmodel.zip'.format(job_id)
            s3.upload_file(zip_path, bucket, s3_key)
            model_urls['savedmodel'] = 'https://{}.s3.amazonaws.com/{}'.format(bucket, s3_key)
            logger.info("Uploaded SavedModel to S3: {}".format(s3_key))
        except Exception as e:
            logger.error("Failed to save/upload SavedModel: {}".format(e))
        # Save classifier spec with model metadata
        try:
            training_metadata = training_result.get('training_metadata', {})
            spec = {
                'type': 'classification',
                'backbone': 'MobileNetV2',
                'image_size': model_manager.image_size,
                'num_classes': int(len(model_manager.id_to_student)),
                'embedding_dim': model_manager.embedding_dim,
                'total_samples': training_metadata.get('total_samples', total_samples),
                'train_samples': training_metadata.get('train_samples'),
                'val_samples': training_metadata.get('val_samples'),
                'phase1_epochs': training_metadata.get('phase1_epochs'),
                'phase2_epochs': training_metadata.get('phase2_epochs'),
                'architecture': 'transfer_learning_mobilenetv2'
            }
            spec_path = '{}/classifier_spec.json'.format(temp_dir)
            with open(spec_path, 'w') as f:
                json.dump(spec, f, indent=2)
            s3_key = 'models/{}/classifier_spec.json'.format(job_id)
            s3.upload_file(spec_path, bucket, s3_key)
            model_urls['classifier_spec'] = 'https://{}.s3.amazonaws.com/{}'.format(bucket, s3_key)
            logger.info("Uploaded classifier_spec.json to S3: {}".format(s3_key))
        except Exception as e:
            logger.error("Failed to write/upload classifier spec: {}".format(e))
        # Compute centroids for each student (fallback for verification)
        try:
            centroids = {}
            # Compute embeddings for each student's genuine signatures
            for class_idx, student_id in model_manager.id_to_student.items():
                # Find the student's data by ID
                student_data = None
                for key, data in processed_data.items():
                    # Match by student ID in the key
                    if str(student_id) in key or key == model_manager.id_to_name.get(student_id):
                        student_data = data
                        break
                
                if student_data and student_data.get('genuine'):
                    imgs = student_data['genuine']
                    if imgs:
                        arr = np.array(imgs, dtype=np.float32)
                        embs = model_manager.embedding_model.predict(arr, verbose=0)
                        # Store centroid by student ID (not class index)
                        centroids[str(student_id)] = np.mean(embs, axis=0).tolist()
            
            if centroids:
                centroids_path = '{}/centroids.json'.format(temp_dir)
                with open(centroids_path, 'w') as f:
                    json.dump(centroids, f, indent=2)
                s3_key = 'models/{}/centroids.json'.format(job_id)
                s3.upload_file(centroids_path, bucket, s3_key)
                model_urls['centroids'] = 'https://{}.s3.amazonaws.com/{}'.format(bucket, s3_key)
                logger.info("Uploaded centroids.json (ID-keyed) to S3: {}".format(s3_key))
        except Exception as e:
            logger.warning("Failed to compute/upload centroids: {}".format(e))
        # Save training results JSON
        try:
            training_results_data = {
                'job_id': job_id,
                'classification_history': training_result.get('classification_history', {}),
                'training_metadata': training_result.get('training_metadata', {}),
                'student_mappings': {
                    'class_index_to_student_id': {str(ci): int(sid) for ci, sid in model_manager.id_to_student.items()},
                    'class_index_to_student_name': {str(ci): model_manager.id_to_name.get(sid, f"Student_{sid}") 
                                                   for ci, sid in model_manager.id_to_student.items()},
                    'num_classes': len(model_manager.id_to_student)
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            results_path = '{}/training_results.json'.format(temp_dir)
            with open(results_path, 'w') as f:
                json.dump(training_results_data, f, indent=2)
            
            s3_key = 'models/{}/training_results.json'.format(job_id)
            s3.upload_file(results_path, bucket, s3_key)
            model_urls['training_results'] = 'https://{}.s3.amazonaws.com/{}'.format(bucket, s3_key)
            logger.info("Uploaded training_results.json to S3: {}".format(s3_key))
        except Exception as e:
            logger.error("Failed to save training results: {}".format(e))
        
        # Extract final metrics
        classification_history = training_result.get('classification_history', {})
        final_accuracy = None
        if 'accuracy' in classification_history:
            accuracies = classification_history['accuracy']
            if accuracies:
                final_accuracy = float(accuracies[-1])
                logger.info("Final training accuracy: {:.4f}".format(final_accuracy))
        final_val_accuracy = None
        if 'val_accuracy' in classification_history:
            val_accuracies = classification_history['val_accuracy']
            if val_accuracies:
                final_val_accuracy = float(val_accuracies[-1])
                logger.info("Final validation accuracy: {:.4f}".format(final_val_accuracy))
        results = {
            'job_id': job_id,
            'student_id': student_id,
            'model_urls': model_urls,
            'accuracy': final_accuracy,
            'val_accuracy': final_val_accuracy,
            'training_metrics': {
                'final_accuracy': final_accuracy,
                'final_val_accuracy': final_val_accuracy,
                'classification_history': classification_history,
                'epochs_trained': len(classification_history.get('loss', [])),
                'total_samples': total_samples,
                'students_count': len(processed_data),
                'preprocessing_stats': {
                    'processed_count': preprocessor.processed_count,
                    'error_count': preprocessor.error_count
                }
            }
        }
        # Ensure JSON-serializable
        def _to_py(o):
            try:
                import numpy as _np
                if isinstance(o, _np.generic):
                    return o.item()
                if isinstance(o, _np.ndarray):
                    return o.tolist()
            except Exception:
                pass
            if isinstance(o, dict):
                return {k: _to_py(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_to_py(v) for v in o]
            return o
        results = _to_py(results)

        results_path = '/tmp/training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        s3.upload_file(results_path, bucket, 'training_results/{}.json'.format(job_id))
        print("Uploaded training results to S3")
        logger.info("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
        logger.info("Job ID: {}".format(job_id))
        logger.info("Final accuracy: {}".format(final_accuracy))
        logger.info("Final validation accuracy: {}".format(final_val_accuracy))
        logger.info("Models uploaded: {} files".format(len(model_urls)))
        logger.info("Total training samples: {}".format(total_samples))
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        logger.error("\n=== TRAINING FAILED ===")
        logger.error("Error: {}".format(str(e)))
        logger.error("Full traceback:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: train_gpu.py <training_data_key> <job_id> <student_id>")
        sys.exit(1)
    training_data_key = sys.argv[1]
    job_id = sys.argv[2]
    student_id = int(sys.argv[3])
    print("Starting GPU training with arguments:")
    print("  Training data key: {}".format(training_data_key))
    print("  Job ID: {}".format(job_id))
    print("  Student ID: {}".format(student_id))
    print("  TensorFlow version: {}".format(tf.__version__))
    train_on_gpu(training_data_key, job_id, student_id)
