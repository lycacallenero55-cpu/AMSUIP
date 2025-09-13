"""
Production-Ready AI System Test Suite
Comprehensive testing for signature verification AI system
"""

import asyncio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import logging
import time
from typing import List, Dict, Tuple
import json

# Import our AI system
from models.signature_embedding_model import SignatureEmbeddingModel
from utils.signature_preprocessing import SignaturePreprocessor, SignatureAugmentation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AISystemTester:
    """
    Comprehensive test suite for the AI signature verification system
    """
    
    def __init__(self):
        self.preprocessor = SignaturePreprocessor(target_size=224)
        self.augmenter = SignatureAugmentation()
        self.ai_model = SignatureEmbeddingModel(max_students=150)
        
    def generate_test_signatures(self, num_students: int = 5, signatures_per_student: int = 10) -> Dict:
        """
        Generate realistic test signatures for testing
        """
        logger.info(f"Generating {num_students} students with {signatures_per_student} signatures each...")
        
        training_data = {}
        
        for student_id in range(num_students):
            student_name = f"student_{student_id}"
            genuine_signatures = []
            forged_signatures = []
            
            # Generate genuine signatures (variations of the same signature)
            base_signature = self._create_base_signature(f"Student {student_id}")
            
            for i in range(signatures_per_student):
                # Create variations of the base signature
                signature = self._create_signature_variation(base_signature, variation_intensity=0.3)
                genuine_signatures.append(signature)
            
            # Generate forged signatures (different signatures)
            for i in range(signatures_per_student // 2):
                forged_signature = self._create_forged_signature(f"Student {student_id}")
                forged_signatures.append(forged_signature)
            
            training_data[student_name] = {
                'genuine': genuine_signatures,
                'forged': forged_signatures
            }
            
            logger.info(f"Generated {len(genuine_signatures)} genuine and {len(forged_signatures)} forged signatures for {student_name}")
        
        return training_data
    
    def _create_base_signature(self, name: str) -> np.ndarray:
        """Create a base signature pattern"""
        # Create a 224x224 white canvas
        img = Image.new('RGB', (224, 224), 'white')
        draw = ImageDraw.Draw(img)
        
        # Create signature-like patterns
        # Main signature line
        start_x, start_y = 50, 100
        end_x, end_y = 180, 120
        
        # Draw signature curves
        points = []
        for i in range(20):
            t = i / 19
            x = start_x + (end_x - start_x) * t
            y = start_y + (end_y - start_y) * t + np.sin(t * np.pi * 3) * 10
            points.append((x, y))
        
        # Draw the signature
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill='black', width=3)
        
        # Add some flourishes
        draw.arc([160, 110, 180, 130], 0, 180, fill='black', width=2)
        draw.arc([40, 90, 60, 110], 0, 180, fill='black', width=2)
        
        return np.array(img)
    
    def _create_signature_variation(self, base_signature: np.ndarray, variation_intensity: float = 0.3) -> np.ndarray:
        """Create a variation of the base signature"""
        # Convert to PIL for processing
        img = Image.fromarray(base_signature)
        
        # Apply random variations
        import random
        
        # Random rotation
        if random.random() < 0.7:
            angle = random.uniform(-variation_intensity * 10, variation_intensity * 10)
            img = img.rotate(angle, fillcolor='white')
        
        # Random scale
        if random.random() < 0.5:
            scale = random.uniform(1 - variation_intensity, 1 + variation_intensity)
            w, h = img.size
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center crop or pad
            if scale > 1:
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                img = img.crop((left, top, left + w, top + h))
            else:
                new_img = Image.new('RGB', (w, h), 'white')
                left = (w - new_w) // 2
                top = (h - new_h) // 2
                new_img.paste(img, (left, top))
                img = new_img
        
        # Random brightness
        if random.random() < 0.4:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(1 - variation_intensity, 1 + variation_intensity)
            img = enhancer.enhance(factor)
        
        return np.array(img)
    
    def _create_forged_signature(self, name: str) -> np.ndarray:
        """Create a forged signature (different pattern)"""
        img = Image.new('RGB', (224, 224), 'white')
        draw = ImageDraw.Draw(img)
        
        # Create a different signature pattern
        start_x, start_y = 60, 110
        end_x, end_y = 170, 130
        
        # Different curve pattern
        points = []
        for i in range(15):
            t = i / 14
            x = start_x + (end_x - start_x) * t
            y = start_y + (end_y - start_y) * t + np.cos(t * np.pi * 2) * 15
            points.append((x, y))
        
        # Draw the forged signature
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill='black', width=2)
        
        # Add different flourishes
        draw.arc([150, 120, 170, 140], 0, 180, fill='black', width=2)
        
        return np.array(img)
    
    def test_preprocessing_pipeline(self, test_signatures: List[np.ndarray]) -> bool:
        """Test the preprocessing pipeline"""
        logger.info("Testing preprocessing pipeline...")
        
        try:
            processed_signatures = []
            for signature in test_signatures:
                processed = self.preprocessor.preprocess_signature(signature)
                processed_signatures.append(processed)
                
                # Validate output
                assert processed.shape == (224, 224, 3), f"Invalid output shape: {processed.shape}"
                assert processed.dtype == np.float32, f"Invalid dtype: {processed.dtype}"
                assert 0 <= processed.min() <= processed.max() <= 1, f"Invalid value range: {processed.min()}-{processed.max()}"
            
            logger.info(f"✅ Preprocessing pipeline test passed: {len(processed_signatures)} signatures processed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Preprocessing pipeline test failed: {e}")
            return False
    
    def test_augmentation_pipeline(self, test_signatures: List[np.ndarray]) -> bool:
        """Test the augmentation pipeline"""
        logger.info("Testing augmentation pipeline...")
        
        try:
            augmented_signatures = []
            for signature in test_signatures:
                # Test genuine augmentation
                aug_genuine = self.augmenter.augment_signature(signature, is_genuine=True)
                augmented_signatures.append(aug_genuine)
                
                # Test forged augmentation
                aug_forged = self.augmenter.augment_signature(signature, is_genuine=False)
                augmented_signatures.append(aug_forged)
                
                # Validate outputs
                assert aug_genuine.shape == signature.shape, f"Invalid augmented shape: {aug_genuine.shape}"
                assert aug_forged.shape == signature.shape, f"Invalid augmented shape: {aug_forged.shape}"
            
            logger.info(f"✅ Augmentation pipeline test passed: {len(augmented_signatures)} augmented signatures created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Augmentation pipeline test failed: {e}")
            return False
    
    def test_ai_model_training(self, training_data: Dict) -> bool:
        """Test AI model training"""
        logger.info("Testing AI model training...")
        
        try:
            # Train the model with a small number of epochs for testing
            start_time = time.time()
            result = self.ai_model.train_models(training_data, epochs=5)  # Reduced epochs for testing
            
            training_time = time.time() - start_time
            
            # Validate training results
            assert 'classification_history' in result, "Missing classification history"
            assert 'authenticity_history' in result, "Missing authenticity history"
            assert 'siamese_history' in result, "Missing siamese history"
            assert 'student_mappings' in result, "Missing student mappings"
            
            # Check that models were created
            assert self.ai_model.embedding_model is not None, "Embedding model not created"
            assert self.ai_model.classification_head is not None, "Classification model not created"
            assert self.ai_model.authenticity_head is not None, "Authenticity model not created"
            assert self.ai_model.siamese_model is not None, "Siamese model not created"
            
            # Check training metrics
            classification_acc = result['classification_history'].get('accuracy', [0])[-1]
            authenticity_acc = result['authenticity_history'].get('accuracy', [0])[-1]
            
            logger.info(f"✅ AI model training test passed:")
            logger.info(f"   - Training time: {training_time:.2f}s")
            logger.info(f"   - Classification accuracy: {classification_acc:.3f}")
            logger.info(f"   - Authenticity accuracy: {authenticity_acc:.3f}")
            logger.info(f"   - Students trained: {len(result['student_mappings']['student_to_id'])}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ AI model training test failed: {e}")
            return False
    
    def test_ai_verification(self, test_signatures: List[np.ndarray]) -> bool:
        """Test AI verification"""
        logger.info("Testing AI verification...")
        
        try:
            verification_results = []
            
            for signature in test_signatures:
                # Preprocess signature
                processed = self.preprocessor.preprocess_signature(signature)
                
                # Verify signature
                result = self.ai_model.verify_signature(processed)
                
                # Validate result structure
                required_keys = [
                    'predicted_student_id', 'predicted_student_name', 
                    'student_confidence', 'is_genuine', 'authenticity_score',
                    'overall_confidence', 'is_unknown', 'embedding'
                ]
                
                for key in required_keys:
                    assert key in result, f"Missing key in verification result: {key}"
                
                # Validate value ranges
                assert 0 <= result['student_confidence'] <= 1, f"Invalid student confidence: {result['student_confidence']}"
                assert 0 <= result['authenticity_score'] <= 1, f"Invalid authenticity score: {result['authenticity_score']}"
                assert 0 <= result['overall_confidence'] <= 1, f"Invalid overall confidence: {result['overall_confidence']}"
                assert isinstance(result['is_genuine'], bool), f"Invalid is_genuine type: {type(result['is_genuine'])}"
                assert isinstance(result['is_unknown'], bool), f"Invalid is_unknown type: {type(result['is_unknown'])}"
                
                verification_results.append(result)
            
            # Analyze results
            avg_confidence = np.mean([r['overall_confidence'] for r in verification_results])
            genuine_count = sum([r['is_genuine'] for r in verification_results])
            
            logger.info(f"✅ AI verification test passed:")
            logger.info(f"   - Verified {len(verification_results)} signatures")
            logger.info(f"   - Average confidence: {avg_confidence:.3f}")
            logger.info(f"   - Genuine signatures detected: {genuine_count}/{len(verification_results)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ AI verification test failed: {e}")
            return False
    
    def test_model_save_load(self) -> bool:
        """Test model saving and loading"""
        logger.info("Testing model save/load...")
        
        try:
            # Save models
            test_path = "/tmp/test_ai_models"
            self.ai_model.save_models(test_path)
            
            # Create new model instance
            new_model = SignatureEmbeddingModel(max_students=150)
            
            # Load models
            success = new_model.load_models(test_path)
            
            assert success, "Model loading failed"
            assert new_model.embedding_model is not None, "Embedding model not loaded"
            assert new_model.classification_head is not None, "Classification model not loaded"
            assert new_model.authenticity_head is not None, "Authenticity model not loaded"
            assert new_model.siamese_model is not None, "Siamese model not loaded"
            
            # Test that loaded model works
            test_signature = self._create_base_signature("Test")
            processed = self.preprocessor.preprocess_signature(test_signature)
            result = new_model.verify_signature(processed)
            
            assert 'predicted_student_id' in result, "Loaded model verification failed"
            
            logger.info("✅ Model save/load test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model save/load test failed: {e}")
            return False
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive test suite"""
        logger.info("🚀 Starting comprehensive AI system test...")
        
        test_results = {
            'preprocessing': False,
            'augmentation': False,
            'training': False,
            'verification': False,
            'save_load': False,
            'overall_success': False
        }
        
        try:
            # Generate test data
            logger.info("📊 Generating test signatures...")
            training_data = self.generate_test_signatures(num_students=3, signatures_per_student=5)
            
            # Extract test signatures
            test_signatures = []
            for student_data in training_data.values():
                test_signatures.extend(student_data['genuine'][:2])  # Take first 2 genuine signatures
            
            # Run tests
            test_results['preprocessing'] = self.test_preprocessing_pipeline(test_signatures)
            test_results['augmentation'] = self.test_augmentation_pipeline(test_signatures)
            test_results['training'] = self.test_ai_model_training(training_data)
            test_results['verification'] = self.test_ai_verification(test_signatures)
            test_results['save_load'] = self.test_model_save_load()
            
            # Overall success
            test_results['overall_success'] = all(test_results.values())
            
            # Summary
            if test_results['overall_success']:
                logger.info("🎉 ALL TESTS PASSED! AI system is production-ready!")
            else:
                failed_tests = [k for k, v in test_results.items() if not v and k != 'overall_success']
                logger.error(f"❌ TESTS FAILED: {failed_tests}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            test_results['overall_success'] = False
            return test_results

def main():
    """Main test function"""
    tester = AISystemTester()
    results = tester.run_comprehensive_test()
    
    # Print final results
    print("\n" + "="*60)
    print("AI SYSTEM TEST RESULTS")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper()}: {status}")
    
    print("="*60)
    
    if results['overall_success']:
        print("🎉 AI SYSTEM IS PRODUCTION-READY!")
        return 0
    else:
        print("❌ AI SYSTEM NEEDS FIXES")
        return 1

if __name__ == "__main__":
    exit(main())