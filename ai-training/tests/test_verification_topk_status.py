import numpy as np
from models.signature_embedding_model import SignatureEmbeddingModel


def test_verify_topk_and_status_with_classifier():
    m = SignatureEmbeddingModel(embedding_dim=16, image_size=64)
    # Build tiny classifier
    m.create_embedding_network()
    m.student_to_id = {"Alice": 0, "Bob": 1}
    m.id_to_student = {0: "Alice", 1: "Bob"}
    m.external_student_id_map = {"Alice": 100, "Bob": 101}
    m.create_classification_head(num_students=2)
    # Create a fake input
    x = np.ones((1, 64, 64, 3), dtype=np.float32)
    # Predict once to warm up
    _ = m.classification_head.predict(x, verbose=0)
    res = m.verify_signature((np.ones((64, 64, 3), dtype=np.float32)))
    assert 'top_k' in res
    assert 'predicted_student_id' in res
    assert isinstance(res['is_unknown'], bool)
