"""
Unit tests for S3-Supabase sync functionality
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from ai_training.utils.s3_supabase_sync import (
    sync_supabase_with_s3,
    fix_student_image_counts,
    count_student_signatures
)


class TestS3SupabaseSync:
    """Test cases for S3-Supabase synchronization"""

    @pytest.fixture
    def mock_s3_client(self):
        """Mock S3 client"""
        client = Mock()
        client.list_objects_v2 = Mock()
        return client

    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client"""
        client = Mock()
        client.table = Mock()
        return client

    @pytest.mark.asyncio
    async def test_count_student_signatures(self, mock_s3_client, mock_supabase_client):
        """Test counting student signatures from S3"""
        # Mock S3 response
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'signatures/student_1/genuine_1.png'},
                {'Key': 'signatures/student_1/genuine_2.png'},
                {'Key': 'signatures/student_1/forged_1.png'},
            ]
        }
        
        # Test counting
        genuine_count, forged_count = await count_student_signatures(
            mock_s3_client, 'test-bucket', 1
        )
        
        assert genuine_count == 2
        assert forged_count == 1

    @pytest.mark.asyncio
    async def test_fix_student_image_counts(self, mock_s3_client, mock_supabase_client):
        """Test fixing student image counts in Supabase"""
        # Mock Supabase response
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = Mock(
            data=[{'id': 1, 'name': 'Test Student'}]
        )
        
        # Mock S3 response
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'signatures/1/genuine_1.png'},
                {'Key': 'signatures/1/genuine_2.png'},
            ]
        }
        
        # Mock update response
        mock_supabase_client.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock(
            data=[{'id': 1, 'genuine_count': 2, 'forged_count': 0}]
        )
        
        # Test fixing counts
        result = await fix_student_image_counts(mock_s3_client, mock_supabase_client, 'test-bucket')
        
        assert result['updated'] == 1
        assert result['errors'] == 0

    @pytest.mark.asyncio
    async def test_sync_supabase_with_s3(self, mock_s3_client, mock_supabase_client):
        """Test full synchronization between Supabase and S3"""
        # Mock Supabase response
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = Mock(
            data=[
                {'id': 1, 'name': 'Student 1', 'genuine_count': 0, 'forged_count': 0},
                {'id': 2, 'name': 'Student 2', 'genuine_count': 0, 'forged_count': 0}
            ]
        )
        
        # Mock S3 responses for different students
        def mock_list_objects(Bucket, Prefix):
            if 'student_1' in Prefix:
                return {
                    'Contents': [
                        {'Key': 'signatures/1/genuine_1.png'},
                        {'Key': 'signatures/1/genuine_2.png'},
                    ]
                }
            elif 'student_2' in Prefix:
                return {
                    'Contents': [
                        {'Key': 'signatures/2/genuine_1.png'},
                    ]
                }
            return {'Contents': []}
        
        mock_s3_client.list_objects_v2.side_effect = mock_list_objects
        
        # Mock update response
        mock_supabase_client.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock(
            data=[{'id': 1, 'genuine_count': 2, 'forged_count': 0}]
        )
        
        # Test synchronization
        result = await sync_supabase_with_s3(mock_s3_client, mock_supabase_client, 'test-bucket')
        
        assert result['students_processed'] == 2
        assert result['students_updated'] == 2
        assert result['errors'] == 0

    @pytest.mark.asyncio
    async def test_sync_handles_s3_errors(self, mock_s3_client, mock_supabase_client):
        """Test that sync handles S3 errors gracefully"""
        # Mock Supabase response
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = Mock(
            data=[{'id': 1, 'name': 'Test Student', 'genuine_count': 0, 'forged_count': 0}]
        )
        
        # Mock S3 error
        mock_s3_client.list_objects_v2.side_effect = Exception("S3 error")
        
        # Test error handling
        result = await sync_supabase_with_s3(mock_s3_client, mock_supabase_client, 'test-bucket')
        
        assert result['students_processed'] == 1
        assert result['students_updated'] == 0
        assert result['errors'] == 1

    @pytest.mark.asyncio
    async def test_sync_handles_supabase_errors(self, mock_s3_client, mock_supabase_client):
        """Test that sync handles Supabase errors gracefully"""
        # Mock Supabase error
        mock_supabase_client.table.return_value.select.return_value.execute.side_effect = Exception("Supabase error")
        
        # Test error handling
        result = await sync_supabase_with_s3(mock_s3_client, mock_supabase_client, 'test-bucket')
        
        assert result['students_processed'] == 0
        assert result['students_updated'] == 0
        assert result['errors'] == 1