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
import tempfile
import shutil
import base64

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Found {} GPU(s), using GPU acceleration".format(len(gpus)))
    except RuntimeError as e:
        print("GPU setup error: {}".format(e))
else:
    print("No GPU found, using CPU")

class SignaturePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.processed_count = 0
        self.error_count = 0
    def preprocess_signature(self, img_data, debug_name="unknown"):
        try:
            img = None
            if isinstance(img_data, str):
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
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
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
    def __init__(self, max_students=150):
        self.max_students = max_students
        self.embedding_dim = 128
        self.student_to_id = {}
        self.id_to_student = {}
        self.embedding_model = None
        self.classification_head = None
        self.siamese_model = None
    def train_models(self, training_data, epochs=25, validation_split=0.2):
        print("Starting model training...")
        all_images = []
        all_labels = []
        print("Processing {} students...".format(len(training_data)))
        for idx, (student_name, data) in enumerate(training_data.items()):
            self.student_to_id[student_name] = idx
            self.id_to_student[idx] = student_name
            genuine_count = len(data.get('genuine', []))
            forged_count = len(data.get('forged', []))
            print("Student {} (ID: {}): {} genuine, {} forged".format(student_name, idx, genuine_count, forged_count))
            for i, img in enumerate(data.get('genuine', [])):
                if img is not None:
                    all_images.append(img)
                    all_labels.append(idx)
                else:
                    print("    Skipping None genuine image {} for {}".format(i, student_name))
            for i, img in enumerate(data.get('forged', [])):
                if img is not None:
                    all_images.append(img)
                    all_labels.append(idx)
                else:
                    print("    Skipping None forged image {} for {}".format(i, student_name))
        print("Total images for training: {}".format(len(all_images)))
        print("Unique student IDs: {}".format(len(set(all_labels))))
        if len(all_images) == 0:
            raise ValueError("No valid training samples found after processing")
        if len(all_images) < 5:
            print("WARNING: Very few samples for training. Results may be poor.")
            validation_split = 0.0
        print("Converting to numpy arrays...")
        X = np.array(all_images)
        y = keras.utils.to_categorical(all_labels, num_classes=len(training_data))
        print("Final training data shape: {}".format(X.shape))
        print("Labels shape: {}".format(y.shape))
        print("Data type: {}, range: [{:.3f}, {:.3f}]".format(X.dtype, float(X.min()), float(X.max())))
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(224, 224, 3)),
            keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2,2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2,2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(training_data), activation='softmax')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model summary:")
        model.summary()
        print("Starting training with validation_split={}".format(validation_split))
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy' if validation_split>0 else 'accuracy', patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss' if validation_split>0 else 'loss', factor=0.5, patience=3, min_lr=0.0001),
        ]
        try:
            if validation_split > 0:
                history = model.fit(X, y, batch_size=min(32, len(X)), epochs=epochs, validation_split=validation_split, callbacks=callbacks, verbose=1)
            else:
                history = model.fit(X, y, batch_size=min(32, len(X)), epochs=epochs, callbacks=callbacks, verbose=1)
        except Exception as e:
            print("Training failed: {}".format(e))
            traceback.print_exc()
            raise
        self.classification_head = model
        self.embedding_model = keras.Model(inputs=model.input, outputs=model.layers[-3].output)
        print("Training completed successfully!")
        return {'classification_history': history.history, 'siamese_history': {}}

def train_on_gpu(training_data_key, job_id, student_id):
    try:
        s3 = boto3.client('s3')
        bucket = os.environ.get('S3_BUCKET', 'signatureai-uploads')
        print("Starting training for job {}".format(job_id))
        print("S3 bucket: {}".format(bucket))
        print("Training data key: {}".format(training_data_key))
        print("Downloading training data from s3://{}/{}".format(bucket, training_data_key))
        try:
            response = s3.get_object(Bucket=bucket, Key=training_data_key)
            training_data_raw = json.loads(response['Body'].read())
            print("Successfully downloaded training data")
        except Exception as e:
            print("Failed to download training data: {}".format(e))
            traceback.print_exc()
            raise
        print("Raw training data contains {} students".format(len(training_data_raw)))
        preprocessor = SignaturePreprocessor(target_size=(224, 224))
        model_manager = SignatureEmbeddingModel(max_students=150)
        print("Processing training data...")
        processed_data = {}
        for student_name, data in training_data_raw.items():
            print("\nProcessing student: {}".format(student_name))
            genuine_images = []
            forged_images = []
            genuine_raw = data.get('genuine', [])
            print("  Found {} genuine images".format(len(genuine_raw)))
            for i, img_data in enumerate(genuine_raw):
                print("    Processing genuine image {}/{}".format(i+1, len(genuine_raw)))
                processed_img = preprocessor.preprocess_signature(img_data, debug_name="{}_genuine_{}".format(student_name, i))
                if processed_img is not None:
                    genuine_images.append(processed_img)
                    print("      \u2713 Successfully processed")
                else:
                    print("      x Failed to process")
            forged_raw = data.get('forged', [])
            print("  Found {} forged images".format(len(forged_raw)))
            for i, img_data in enumerate(forged_raw):
                print("    Processing forged image {}/{}".format(i+1, len(forged_raw)))
                processed_img = preprocessor.preprocess_signature(img_data, debug_name="{}_forged_{}".format(student_name, i))
                if processed_img is not None:
                    forged_images.append(processed_img)
                    print("      \u2713 Successfully processed")
                else:
                    print("      x Failed to process")
            processed_data[student_name] = {'genuine': genuine_images, 'forged': forged_images}
            total_processed = len(genuine_images) + len(forged_images)
            total_raw = len(genuine_raw) + len(forged_raw)
            success_rate = (total_processed / total_raw * 100) if total_raw > 0 else 0
            print("  Final: {} genuine, {} forged".format(len(genuine_images), len(forged_images)))
            print("  Success rate: {}/{} ({:.1f}%)".format(total_processed, total_raw, success_rate))
        print("\n=== PREPROCESSING SUMMARY ===")
        print("Total images processed successfully: {}".format(preprocessor.processed_count))
        print("Total processing errors: {}".format(preprocessor.error_count))
        total_samples = sum(len(d['genuine']) + len(d['forged']) for d in processed_data.values())
        print("Total processed samples available for training: {}".format(total_samples))
        if total_samples == 0:
            raise ValueError("No valid training samples found after processing")
        validation_split = 0.0 if total_samples < 5 else 0.2
        print("\n=== STARTING MODEL TRAINING ===")
        training_result = model_manager.train_models(processed_data, epochs=25, validation_split=validation_split)
        print("Training completed! Saving models...")
        temp_dir = '/tmp/{}_models'.format(job_id)
        os.makedirs(temp_dir, exist_ok=True)
        model_manager.save_models('{}/signature_model'.format(temp_dir))
        model_files = ['embedding','classification']
        model_urls = {}
        for model_type in model_files:
            file_path = '{}/signature_model_{}.keras'.format(temp_dir, model_type)
            if os.path.exists(file_path):
                s3_key = 'models/{}/{}.keras'.format(job_id, model_type)
                s3.upload_file(file_path, bucket, s3_key)
                model_urls[model_type] = 'https://{}.s3.amazonaws.com/{}'.format(bucket, s3_key)
                print("Uploaded {} model to S3: {}".format(model_type, s3_key))
            else:
                print("WARNING: {} model file not found: {}".format(model_type, file_path))
        mappings_path = '{}/signature_model_mappings.json'.format(temp_dir)
        if os.path.exists(mappings_path):
            s3_key = 'models/{}/mappings.json'.format(job_id)
            s3.upload_file(mappings_path, bucket, s3_key)
            model_urls['mappings'] = 'https://{}.s3.amazonaws.com/{}'.format(bucket, s3_key)
            print("Uploaded mappings to S3: {}".format(s3_key))
        classification_history = training_result.get('classification_history', {})
        final_accuracy = None
        if 'accuracy' in classification_history:
            accuracies = classification_history['accuracy']
            if accuracies:
                final_accuracy = float(accuracies[-1])
                print("Final training accuracy: {:.4f}".format(final_accuracy))
        final_val_accuracy = None
        if 'val_accuracy' in classification_history:
            val_accuracies = classification_history['val_accuracy']
            if val_accuracies:
                final_val_accuracy = float(val_accuracies[-1])
                print("Final validation accuracy: {:.4f}".format(final_val_accuracy))
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
        results_path = '/tmp/training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        s3.upload_file(results_path, bucket, 'training_results/{}.json'.format(job_id))
        print("Uploaded training results to S3")
        print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
        print("Job ID: {}".format(job_id))
        print("Final accuracy: {}".format(final_accuracy))
        print("Models uploaded: {} files".format(len(model_urls)))
        print("Total training samples: {}".format(total_samples))
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print("\n=== TRAINING FAILED ===")
        print("Error: {}".format(str(e)))
        print("Full traceback:")
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

