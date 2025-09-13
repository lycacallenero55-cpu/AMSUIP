"""
S3-Supabase Synchronization Utilities
Handles synchronization between S3 storage and Supabase database records
"""

import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import asyncio

from models.database import db_manager
from utils.s3_storage import object_exists, count_objects_with_prefix
from config import settings

logger = logging.getLogger(__name__)

class S3SupabaseSync:
    """Handles synchronization between S3 and Supabase"""
    
    def __init__(self):
        self.sync_stats = {
            'total_records_checked': 0,
            'missing_s3_objects': 0,
            'stale_db_records': 0,
            'records_updated': 0,
            'records_deleted': 0,
            'errors': 0
        }
    
    async def sync_supabase_with_s3(self, dry_run: bool = True) -> Dict:
        """
        Synchronize Supabase records with S3 objects
        
        Args:
            dry_run: If True, only report issues without making changes
            
        Returns:
            Dictionary with sync statistics
        """
        logger.info(f"Starting S3-Supabase sync (dry_run={dry_run})")
        self.sync_stats = {k: 0 for k in self.sync_stats.keys()}
        
        try:
            # Get all signature records from Supabase
            all_signatures = await db_manager.list_all_signatures()
            self.sync_stats['total_records_checked'] = len(all_signatures)
            
            logger.info(f"Checking {len(all_signatures)} signature records")
            
            # Check each signature record
            for signature in all_signatures:
                await self._check_signature_record(signature, dry_run)
            
            # Check for orphaned S3 objects (optional - can be expensive)
            if not dry_run:
                await self._check_orphaned_s3_objects()
            
            logger.info(f"Sync completed: {self.sync_stats}")
            return self.sync_stats
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.sync_stats['errors'] += 1
            raise
    
    async def _check_signature_record(self, signature: Dict, dry_run: bool):
        """Check a single signature record against S3"""
        try:
            s3_key = signature.get('s3_key')
            s3_url = signature.get('s3_url')
            record_id = signature.get('id')
            student_id = signature.get('student_id')
            
            if not s3_key:
                logger.warning(f"Record {record_id} has no s3_key")
                self.sync_stats['errors'] += 1
                return
            
            # Check if S3 object exists
            exists = object_exists(s3_key)
            
            if not exists:
                logger.warning(f"S3 object missing for record {record_id}: {s3_key}")
                self.sync_stats['missing_s3_objects'] += 1
                
                if not dry_run:
                    # Mark as missing_images=true or delete
                    await self._handle_missing_s3_object(record_id, student_id, s3_key)
            else:
                logger.debug(f"S3 object exists for record {record_id}: {s3_key}")
                
        except Exception as e:
            logger.error(f"Error checking signature record {signature.get('id')}: {e}")
            self.sync_stats['errors'] += 1
    
    async def _handle_missing_s3_object(self, record_id: int, student_id: int, s3_key: str):
        """Handle missing S3 object by updating or deleting DB record"""
        try:
            # Option 1: Mark as missing_images=true (if column exists)
            # Option 2: Delete the record entirely
            
            # For now, delete the record since it's orphaned
            success = await db_manager.delete_signature(record_id)
            
            if success:
                logger.info(f"Deleted orphaned record {record_id} for student {student_id}")
                self.sync_stats['records_deleted'] += 1
            else:
                logger.error(f"Failed to delete record {record_id}")
                self.sync_stats['errors'] += 1
                
        except Exception as e:
            logger.error(f"Error handling missing S3 object for record {record_id}: {e}")
            self.sync_stats['errors'] += 1
    
    async def _check_orphaned_s3_objects(self):
        """Check for S3 objects that don't have corresponding DB records"""
        try:
            # This is expensive, so we'll do it in batches
            # For now, we'll skip this to avoid high costs
            logger.info("Skipping orphaned S3 object check (expensive operation)")
            pass
            
        except Exception as e:
            logger.error(f"Error checking orphaned S3 objects: {e}")
            self.sync_stats['errors'] += 1
    
    async def get_students_with_missing_images(self) -> List[Dict]:
        """Get list of students who have DB records but missing S3 images"""
        try:
            all_signatures = await db_manager.list_all_signatures()
            missing_images = []
            
            for signature in all_signatures:
                s3_key = signature.get('s3_key')
                if s3_key and not object_exists(s3_key):
                    missing_images.append({
                        'student_id': signature.get('student_id'),
                        'record_id': signature.get('id'),
                        's3_key': s3_key,
                        'label': signature.get('label')
                    })
            
            return missing_images
            
        except Exception as e:
            logger.error(f"Error getting students with missing images: {e}")
            return []
    
    async def fix_student_image_counts(self, student_id: int) -> Dict:
        """Fix image counts for a specific student"""
        try:
            # Get actual counts from S3
            genuine_count, forged_count = count_student_signatures(student_id)
            
            # Get current DB records
            signatures = await db_manager.list_student_signatures(student_id)
            
            # Remove records for missing S3 objects
            valid_records = []
            for sig in signatures:
                s3_key = sig.get('s3_key')
                if s3_key and object_exists(s3_key):
                    valid_records.append(sig)
                else:
                    # Delete invalid record
                    await db_manager.delete_signature(sig.get('id'))
            
            return {
                'student_id': student_id,
                's3_genuine_count': genuine_count,
                's3_forged_count': forged_count,
                'db_records_after_cleanup': len(valid_records),
                'records_deleted': len(signatures) - len(valid_records)
            }
            
        except Exception as e:
            logger.error(f"Error fixing image counts for student {student_id}: {e}")
            return {'error': str(e)}

# Global sync instance
sync_manager = S3SupabaseSync()

async def sync_supabase_with_s3(dry_run: bool = True) -> Dict:
    """
    Main sync function - synchronizes Supabase with S3
    
    Args:
        dry_run: If True, only report issues without making changes
        
    Returns:
        Dictionary with sync statistics
    """
    return await sync_manager.sync_supabase_with_s3(dry_run)

async def get_students_with_missing_images() -> List[Dict]:
    """Get students with missing S3 images"""
    return await sync_manager.get_students_with_missing_images()

async def fix_student_image_counts(student_id: int) -> Dict:
    """Fix image counts for a specific student"""
    return await sync_manager.fix_student_image_counts(student_id)
