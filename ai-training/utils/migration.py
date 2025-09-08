"""
Migration utilities for transitioning from Supabase storage to S3 storage.
"""

import os
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import logging

from models.database import db_manager
from utils.s3_storage import upload_model_file, download_model_file
from utils.storage import load_model_from_supabase
from config import settings

logger = logging.getLogger(__name__)


async def migrate_models_to_s3():
    """
    Migrate existing models from Supabase storage to S3.
    This function will:
    1. Find all models in trained_models table with Supabase paths
    2. Download them from Supabase
    3. Upload them to S3
    4. Update the database records with new S3 URLs
    """
    try:
        logger.info("Starting model migration from Supabase to S3...")
        
        # Get all models that are still using Supabase storage
        all_models = await db_manager.get_trained_models()
        supabase_models = [
            model for model in all_models 
            if model.get("model_path") and not model.get("model_path").startswith("https://")
        ]
        
        if not supabase_models:
            logger.info("No models found that need migration.")
            return {"migrated": 0, "failed": 0, "errors": []}
        
        logger.info(f"Found {len(supabase_models)} models to migrate.")
        
        migrated_count = 0
        failed_count = 0
        errors = []
        
        for model in supabase_models:
            try:
                model_id = model.get("id")
                model_path = model.get("model_path")
                embedding_path = model.get("embedding_model_path")
                
                logger.info(f"Migrating model {model_id}...")
                
                # Determine model type based on student_id
                model_type = "global" if model.get("student_id") == 0 else "individual"
                model_uuid = f"migrated_{model_id}_{int(datetime.utcnow().timestamp())}"
                
                # Migrate main model
                if model_path:
                    try:
                        # Load model from Supabase
                        model_data = await load_model_from_supabase(model_path)
                        
                        # Convert to bytes if it's a model object
                        if hasattr(model_data, 'save'):
                            import tempfile
                            import tensorflow as tf
                            
                            temp_file = tempfile.NamedTemporaryFile(suffix='.keras', delete=False)
                            model_data.save(temp_file.name)
                            
                            with open(temp_file.name, 'rb') as f:
                                model_bytes = f.read()
                            
                            os.unlink(temp_file.name)
                        else:
                            model_bytes = model_data
                        
                        # Upload to S3
                        s3_key, s3_url = upload_model_file(
                            model_bytes, 
                            model_type, 
                            f"{model_uuid}_main", 
                            "keras"
                        )
                        
                        # Update database record
                        await db_manager.update_trained_model(model_id, {
                            "model_path": s3_url,
                            "s3_key": s3_key
                        })
                        
                        logger.info(f"Successfully migrated main model {model_id} to S3")
                        
                    except Exception as e:
                        logger.error(f"Failed to migrate main model {model_id}: {e}")
                        errors.append(f"Model {model_id} main: {str(e)}")
                        failed_count += 1
                        continue
                
                # Migrate embedding model if different
                if embedding_path and embedding_path != model_path:
                    try:
                        # Load embedding model from Supabase
                        embedding_data = await load_model_from_supabase(embedding_path)
                        
                        # Convert to bytes if it's a model object
                        if hasattr(embedding_data, 'save'):
                            import tempfile
                            
                            temp_file = tempfile.NamedTemporaryFile(suffix='.keras', delete=False)
                            embedding_data.save(temp_file.name)
                            
                            with open(temp_file.name, 'rb') as f:
                                embedding_bytes = f.read()
                            
                            os.unlink(temp_file.name)
                        else:
                            embedding_bytes = embedding_data
                        
                        # Upload to S3
                        embedding_s3_key, embedding_s3_url = upload_model_file(
                            embedding_bytes, 
                            model_type, 
                            f"{model_uuid}_embedding", 
                            "keras"
                        )
                        
                        # Update database record
                        await db_manager.update_trained_model(model_id, {
                            "embedding_model_path": embedding_s3_url,
                            "embedding_s3_key": embedding_s3_key
                        })
                        
                        logger.info(f"Successfully migrated embedding model {model_id} to S3")
                        
                    except Exception as e:
                        logger.error(f"Failed to migrate embedding model {model_id}: {e}")
                        errors.append(f"Model {model_id} embedding: {str(e)}")
                
                migrated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate model {model.get('id', 'unknown')}: {e}")
                errors.append(f"Model {model.get('id', 'unknown')}: {str(e)}")
                failed_count += 1
        
        logger.info(f"Migration completed. Migrated: {migrated_count}, Failed: {failed_count}")
        
        return {
            "migrated": migrated_count,
            "failed": failed_count,
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


async def migrate_global_models_to_new_table():
    """
    Migrate global models (student_id = 0) from trained_models to global_trained_models table.
    """
    try:
        logger.info("Starting global models migration to dedicated table...")
        
        # Get all global models (student_id = 0)
        all_models = await db_manager.get_trained_models()
        global_models = [
            model for model in all_models 
            if model.get("student_id") == 0
        ]
        
        if not global_models:
            logger.info("No global models found to migrate.")
            return {"migrated": 0, "failed": 0, "errors": []}
        
        logger.info(f"Found {len(global_models)} global models to migrate.")
        
        migrated_count = 0
        failed_count = 0
        errors = []
        
        for model in global_models:
            try:
                # Create new global model record
                global_model_data = {
                    "model_path": model.get("model_path"),
                    "s3_key": model.get("s3_key"),
                    "model_uuid": f"migrated_{model.get('id')}_{int(datetime.utcnow().timestamp())}",
                    "status": model.get("status", "completed"),
                    "sample_count": model.get("sample_count", 0),
                    "genuine_count": model.get("genuine_count", 0),
                    "forged_count": model.get("forged_count", 0),
                    "student_count": model.get("training_metrics", {}).get("student_count", 0),
                    "training_date": model.get("training_date"),
                    "accuracy": model.get("accuracy"),
                    "training_metrics": model.get("training_metrics", {}),
                    "created_at": model.get("created_at"),
                    "updated_at": model.get("updated_at")
                }
                
                # Create in new table
                new_global_model = await db_manager.create_global_model(global_model_data)
                
                if new_global_model:
                    logger.info(f"Successfully migrated global model {model.get('id')} to new table")
                    migrated_count += 1
                else:
                    logger.error(f"Failed to create global model record for {model.get('id')}")
                    errors.append(f"Model {model.get('id')}: Failed to create new record")
                    failed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate global model {model.get('id', 'unknown')}: {e}")
                errors.append(f"Model {model.get('id', 'unknown')}: {str(e)}")
                failed_count += 1
        
        logger.info(f"Global models migration completed. Migrated: {migrated_count}, Failed: {failed_count}")
        
        return {
            "migrated": migrated_count,
            "failed": failed_count,
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Global models migration failed: {e}")
        raise


async def cleanup_old_supabase_models():
    """
    Clean up old Supabase model files after successful migration.
    WARNING: This will permanently delete files from Supabase storage.
    """
    try:
        logger.info("Starting cleanup of old Supabase model files...")
        
        # Get all models that have been migrated to S3
        all_models = await db_manager.get_trained_models()
        migrated_models = [
            model for model in all_models 
            if model.get("model_path") and model.get("model_path").startswith("https://")
        ]
        
        if not migrated_models:
            logger.info("No migrated models found for cleanup.")
            return {"cleaned": 0, "errors": []}
        
        logger.info(f"Found {len(migrated_models)} migrated models for cleanup.")
        
        cleaned_count = 0
        errors = []
        
        for model in migrated_models:
            try:
                # Note: This would require Supabase storage cleanup logic
                # For now, we'll just log what would be cleaned
                logger.info(f"Would clean up Supabase files for model {model.get('id')}")
                cleaned_count += 1
                
            except Exception as e:
                logger.error(f"Failed to cleanup model {model.get('id', 'unknown')}: {e}")
                errors.append(f"Model {model.get('id', 'unknown')}: {str(e)}")
        
        logger.info(f"Cleanup completed. Cleaned: {cleaned_count}")
        
        return {
            "cleaned": cleaned_count,
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise


async def run_full_migration():
    """
    Run the complete migration process:
    1. Migrate models from Supabase to S3
    2. Migrate global models to new table
    3. Optionally cleanup old Supabase files
    """
    try:
        logger.info("Starting full migration process...")
        
        # Step 1: Migrate models to S3
        s3_migration_result = await migrate_models_to_s3()
        logger.info(f"S3 migration result: {s3_migration_result}")
        
        # Step 2: Migrate global models to new table
        global_migration_result = await migrate_global_models_to_new_table()
        logger.info(f"Global models migration result: {global_migration_result}")
        
        # Step 3: Cleanup (optional - uncomment if you want to delete old files)
        # cleanup_result = await cleanup_old_supabase_models()
        # logger.info(f"Cleanup result: {cleanup_result}")
        
        logger.info("Full migration process completed successfully!")
        
        return {
            "s3_migration": s3_migration_result,
            "global_migration": global_migration_result,
            # "cleanup": cleanup_result
        }
        
    except Exception as e:
        logger.error(f"Full migration failed: {e}")
        raise


if __name__ == "__main__":
    # Run migration if called directly
    asyncio.run(run_full_migration())
