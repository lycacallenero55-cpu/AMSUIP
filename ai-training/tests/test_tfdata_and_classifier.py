import numpy as np
from models.global_signature_model import GlobalSignatureVerificationModel
from utils.tfdata import make_tfdata_from_numpy, split_train_val


def _toy_dataset(num_classes=3, per_class=6, image_size=224):
    rng = np.random.default_rng(123)
    data = {}
    for c in range(num_classes):
        imgs = []
        for _ in range(per_class):
            img = rng.random((image_size, image_size, 3), dtype=np.float32)
            imgs.append(img)
        data[c + 1000] = {"genuine_images": imgs, "forged_images": []}
    return data


def test_global_classifier_trains_minimal():
    data = _toy_dataset(num_classes=3, per_class=4)
    model = GlobalSignatureVerificationModel()
    hist = model.train_global_classifier(data, epochs=1)
    assert 'accuracy' in hist.history

