#!/usr/bin/env python3
"""
Unit Test for S3-Supabase Sync
Tests the sync_supabase_with_s3() function and related utilities
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.s3_supabase_sync import S3SupabaseSync
from models.database import DatabaseManager
from utils.s3_storage import S3StorageManager

async def test_s3_supabase_sync():
    """Test S3-Supabase synchronization"""
    print("🔄 Testing S3-Supabase Sync")
    print("=" * 40)
    
    try:
        # Initialize managers
        db_manager = DatabaseManager()
        s3_manager = S3StorageManager()
        sync_manager = S3SupabaseSync()
        
        # Test 1: Check for missing S3 objects
        print("1. Testing missing S3 objects detection...")
        try:
            result = await sync_manager.sync_supabase_with_s3()
            print(f"   📊 Sync result: {result}")
            print("   ✅ Missing S3 objects check completed.")
        except Exception as e:
            print(f"   ❌ Missing S3 objects check failed: {e}")
            return False
        
        # Test 2: Test orphaned records cleanup
        print("2. Testing orphaned records cleanup...")
        try:
            # First, create a test signature record
            test_signature = {
                "student_id": 999,  # Non-existent student
                "s3_key": "test/orphaned_signature.png",
                "filename": "orphaned_signature.png",
                "is_genuine": True,
                "upload_timestamp": "now()"
            }
            
            # Create the record
            success = await db_manager.create_signature(test_signature)
            if success:
                print("   📝 Created test orphaned record.")
                
                # Run cleanup (dry run first)
                cleanup_result = await sync_manager.cleanup_orphaned_records(dry_run=True)
                print(f"   📊 Cleanup result (dry run): {cleanup_result}")
                
                # Clean up the test record
                await db_manager.delete_student_signatures(999)
                print("   🧹 Cleaned up test record.")
                
                print("   ✅ Orphaned records cleanup test completed.")
            else:
                print("   ⚠️  Could not create test record, skipping cleanup test.")
                
        except Exception as e:
            print(f"   ❌ Orphaned records cleanup test failed: {e}")
            return False
        
        # Test 3: Test S3 object existence check
        print("3. Testing S3 object existence check...")
        try:
            from utils.s3_supabase_sync import object_exists
            
            # Test with non-existent object
            exists = object_exists("test/non_existent_file.png")
            print(f"   📊 Non-existent object exists: {exists}")
            
            # Test with a real object (if any exist)
            signatures = await db_manager.list_all_signatures()
            if signatures:
                test_s3_key = signatures[0].get('s3_key')
                if test_s3_key:
                    exists = object_exists(test_s3_key)
                    print(f"   📊 Real object exists: {exists}")
            
            print("   ✅ S3 object existence check completed.")
            
        except Exception as e:
            print(f"   ❌ S3 object existence check failed: {e}")
            return False
        
        print("=" * 40)
        print("🎉 All sync tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Sync test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_s3_supabase_sync())
    sys.exit(0 if success else 1)