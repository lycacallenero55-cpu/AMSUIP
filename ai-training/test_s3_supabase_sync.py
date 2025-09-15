#!/usr/bin/env python3
"""
Unit tests for S3-Supabase synchronization functionality
Tests the sync_supabase_with_s3 function and related utilities
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json

# Mock the database and S3 dependencies
@pytest.fixture
def mock_db_manager():
    """Mock database manager for testing"""
    manager = Mock()
    manager.list_all_signatures = AsyncMock()
    manager.delete_signature = AsyncMock()
    manager.update_signature = AsyncMock()
    return manager

@pytest.fixture
def mock_s3_storage():
    """Mock S3 storage for testing"""
    storage = Mock()
    storage.object_exists = Mock()
    storage.delete_object = Mock()
    return storage

@pytest.fixture
def sample_signatures():
    """Sample signature data for testing"""
    return [
        {
            'id': 1,
            'student_id': 101,
            's3_key': 'signatures/student_101/sig_1.jpg',
            's3_url': 'https://bucket.s3.amazonaws.com/signatures/student_101/sig_1.jpg',
            'label': 'genuine',
            'created_at': '2024-01-01T00:00:00Z'
        },
        {
            'id': 2,
            'student_id': 101,
            's3_key': 'signatures/student_101/sig_2.jpg',
            's3_url': 'https://bucket.s3.amazonaws.com/signatures/student_101/sig_2.jpg',
            'label': 'genuine',
            'created_at': '2024-01-01T00:00:00Z'
        },
        {
            'id': 3,
            'student_id': 102,
            's3_key': 'signatures/student_102/sig_1.jpg',
            's3_url': 'https://bucket.s3.amazonaws.com/signatures/student_102/sig_1.jpg',
            'label': 'genuine',
            'created_at': '2024-01-01T00:00:00Z'
        }
    ]

class TestS3SupabaseSync:
    """Test cases for S3-Supabase synchronization"""
    
    @pytest.mark.asyncio
    async def test_sync_dry_run(self, mock_db_manager, mock_s3_storage, sample_signatures):
        """Test sync in dry run mode (no actual changes)"""
        # Setup mocks
        mock_db_manager.list_all_signatures.return_value = sample_signatures
        mock_s3_storage.object_exists.return_value = True
        
        # Import the sync function
        from utils.s3_supabase_sync import sync_supabase_with_s3_enhanced
        
        # Run sync in dry run mode
        result = await sync_supabase_with_s3_enhanced(dry_run=True)
        
        # Verify results
        assert result['total_records_checked'] == 3
        assert result['missing_s3_objects'] == 0
        assert result['records_deleted'] == 0
        assert result['errors'] == 0
        
        # Verify no actual changes were made
        mock_db_manager.delete_signature.assert_not_called()
        mock_db_manager.update_signature.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_sync_with_missing_s3_objects(self, mock_db_manager, mock_s3_storage, sample_signatures):
        """Test sync when S3 objects are missing"""
        # Setup mocks - first signature missing from S3
        mock_db_manager.list_all_signatures.return_value = sample_signatures
        mock_s3_storage.object_exists.side_effect = lambda key: key != 'signatures/student_101/sig_1.jpg'
        
        # Import the sync function
        from utils.s3_supabase_sync import sync_supabase_with_s3_enhanced
        
        # Run sync in live mode
        result = await sync_supabase_with_s3_enhanced(dry_run=False)
        
        # Verify results
        assert result['total_records_checked'] == 3
        assert result['missing_s3_objects'] == 1
        assert result['records_deleted'] == 1
        assert result['errors'] == 0
        
        # Verify database was updated
        mock_db_manager.delete_signature.assert_called_once_with(1)
    
    @pytest.mark.asyncio
    async def test_sync_with_database_error(self, mock_db_manager, mock_s3_storage, sample_signatures):
        """Test sync when database operations fail"""
        # Setup mocks - database error
        mock_db_manager.list_all_signatures.side_effect = Exception("Database connection failed")
        
        # Import the sync function
        from utils.s3_supabase_sync import sync_supabase_with_s3_enhanced
        
        # Run sync and expect exception
        with pytest.raises(Exception, match="Database connection failed"):
            await sync_supabase_with_s3_enhanced(dry_run=True)
    
    @pytest.mark.asyncio
    async def test_ensure_atomic_operations(self, mock_db_manager):
        """Test atomic operations check"""
        from utils.s3_supabase_sync import ensure_atomic_operations
        
        # Test successful connection
        mock_db_manager.test_connection.return_value = True
        result = await ensure_atomic_operations()
        assert result is True
        
        # Test failed connection
        mock_db_manager.test_connection.return_value = False
        result = await ensure_atomic_operations()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_fix_student_image_counts(self, mock_db_manager, sample_signatures):
        """Test fixing student image counts"""
        from utils.s3_supabase_sync import fix_student_image_counts
        
        # Setup mocks
        student_signatures = [s for s in sample_signatures if s['student_id'] == 101]
        mock_db_manager.list_signatures_by_student.return_value = student_signatures
        mock_db_manager.delete_signature.return_value = True
        
        # Test fixing counts
        result = await fix_student_image_counts(101)
        
        # Verify results
        assert result['student_id'] == 101
        assert result['records_deleted'] == 2
        assert result['success'] is True
        
        # Verify database calls
        mock_db_manager.list_signatures_by_student.assert_called_once_with(101)
        assert mock_db_manager.delete_signature.call_count == 2

class TestAtomicOperations:
    """Test atomic operations for S3-DB consistency"""
    
    @pytest.mark.asyncio
    async def test_atomic_model_save(self, mock_db_manager, mock_s3_storage):
        """Test atomic saving of model to S3 then DB"""
        from utils.s3_supabase_sync import ensure_atomic_operations
        
        # Mock successful atomic operations
        mock_db_manager.test_connection.return_value = True
        
        # Test atomic operations check
        result = await ensure_atomic_operations()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_atomic_operations_failure(self, mock_db_manager):
        """Test when atomic operations are not available"""
        from utils.s3_supabase_sync import ensure_atomic_operations
        
        # Mock failed connection
        mock_db_manager.test_connection.side_effect = Exception("Connection failed")
        
        # Test atomic operations check
        result = await ensure_atomic_operations()
        assert result is False

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])