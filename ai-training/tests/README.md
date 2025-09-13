# AI Training System Tests

This directory contains comprehensive tests for the AI signature verification system.

## Test Structure

### Unit Tests (`test_s3_supabase_sync.py`)
- **Purpose**: Test individual components in isolation
- **Coverage**: S3-Supabase synchronization functions
- **Key Tests**:
  - `count_student_signatures()`: Counts genuine and forged signatures from S3
  - `fix_student_image_counts()`: Updates Supabase with correct counts
  - `sync_supabase_with_s3()`: Full synchronization workflow
  - Error handling for S3 and Supabase failures

### Integration Tests (`test_training_integration.py`)
- **Purpose**: Test component interactions and data flow
- **Coverage**: Training pipeline, data augmentation, model architecture
- **Key Tests**:
  - Data augmentation pipeline creates diverse samples
  - Model architecture is correct (MobileNetV2 base)
  - Training with small datasets (3-5 samples)
  - Model saving and loading
  - Adaptive batch size calculation
  - Training metrics logging (CSV)
  - Verification with trained models

### End-to-End Tests (`test_e2e_workflow.py`)
- **Purpose**: Test complete user workflows
- **Coverage**: Full API workflows from upload to verification
- **Key Tests**:
  - Upload → Train → Verify workflow
  - Unknown signature returns "no_match"
  - Identify signature workflow
  - Training with small datasets
  - Error handling for invalid inputs
  - Data augmentation in training

## Running Tests

### Prerequisites
```bash
pip install -r requirements-test.txt
```

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Test Types
```bash
# Unit tests only
python run_tests.py --type unit

# Integration tests only
python run_tests.py --type integration

# End-to-end tests only
python run_tests.py --type e2e
```

### Run with Coverage
```bash
python run_tests.py --coverage
```

### Run with Verbose Output
```bash
python run_tests.py --verbose
```

## Test Configuration

### Environment Variables
Tests use the following environment variables (set in `conftest.py`):
- `ENVIRONMENT=test`
- `S3_BUCKET_NAME=test-bucket`
- `SUPABASE_URL=https://test.supabase.co`
- `SUPABASE_KEY=test-key`

### Mock Services
Tests use mocked versions of:
- S3 client (`mock_s3_client`)
- Supabase client (`mock_supabase_client`)
- AI models (`mock_model`)

## Test Data

### Sample Signature Data
Tests use generated signature images with:
- Size: 28x28 pixels (grayscale)
- Simple patterns (horizontal and vertical lines)
- Random noise for variation

### Test Scenarios
1. **Small Dataset Training**: 3-5 samples per student
2. **Unknown Signature**: Returns "no_match" with low confidence
3. **Known Signature**: Returns correct student with high confidence
4. **Error Handling**: Invalid inputs, missing files, service failures

## Coverage Goals

- **Unit Tests**: 90%+ coverage of utility functions
- **Integration Tests**: 80%+ coverage of training pipeline
- **E2E Tests**: 100% coverage of API endpoints

## Continuous Integration

Tests are designed to run in CI/CD pipelines:
- No external dependencies (all services mocked)
- Fast execution (< 30 seconds for full suite)
- Deterministic results
- Clear failure reporting

## Debugging Tests

### Run Single Test
```bash
python -m pytest tests/test_s3_supabase_sync.py::TestS3SupabaseSync::test_count_student_signatures -v
```

### Run with Debug Output
```bash
python -m pytest tests/test_training_integration.py -v -s
```

### Check Test Coverage
```bash
python run_tests.py --coverage
# Open htmlcov/index.html in browser
```

## Adding New Tests

### Unit Test Template
```python
def test_function_name(self, mock_dependencies):
    """Test description"""
    # Arrange
    input_data = "test input"
    expected_output = "expected result"
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output
```

### Integration Test Template
```python
def test_component_interaction(self, sample_data):
    """Test component interaction"""
    # Arrange
    model = SignatureEmbeddingModel()
    data = sample_data['images']
    
    # Act
    result = model.train_classification_only(data, labels)
    
    # Assert
    assert result.history['loss'][-1] < result.history['loss'][0]
```

### E2E Test Template
```python
def test_api_workflow(self, client, sample_data):
    """Test complete API workflow"""
    # Arrange
    test_image = sample_data['images'][0]
    
    # Act
    response = client.post("/api/endpoint", files={"file": test_image})
    
    # Assert
    assert response.status_code == 200
    assert response.json()["success"] == True
```