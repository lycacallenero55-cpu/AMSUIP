import json
import os
import numpy as np
from tensorflow import keras
from utils.artifacts import package_global_classifier_artifacts, build_classifier_spec


def _dummy_model(num_classes=3, image_size=64):
    inp = keras.Input((image_size, image_size, 3))
    x = keras.layers.Rescaling(1.0/255.0)(inp)
    x = keras.layers.Conv2D(8, 3, activation='relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(num_classes, activation='softmax')(x)
    m = keras.Model(inp, out)
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def test_id_first_mappings_round_trip(tmp_path):
    model_uuid = "uuid-test"
    base_dir = tmp_path.as_posix()
    model = _dummy_model()
    mappings = {
        "class_index_to_student_id": {"0": 100, "1": 101},
        "class_index_to_student_name": {"0": "Alice", "1": "Bob"},
        "student_id_to_class_index": {"100": 0, "101": 1},
    }
    spec = build_classifier_spec(2, 64)
    artifacts = package_global_classifier_artifacts(
        model_uuid,
        base_dir,
        model,
        None,
        mappings,
        None,
        {"final_accuracy": 0.0},
        spec,
    )
    mpath = artifacts["mappings_path"]
    with open(mpath, 'r') as f:
        data = json.load(f)
    assert data["class_index_to_student_id"]["0"] == 100
    assert data["student_id_to_class_index"]["100"] == 0


def test_artifact_packaging_integrity(tmp_path):
    model = _dummy_model()
    model_uuid = "uuid-artifacts"
    artifacts = package_global_classifier_artifacts(
        model_uuid,
        tmp_path.as_posix(),
        model,
        None,
        {"class_index_to_student_id": {"0": 1}, "class_index_to_student_name": {"0": "X"}, "student_id_to_class_index": {"1": 0}},
        None,
        {"final_accuracy": 0.0},
        build_classifier_spec(1, 64),
    )
    expected = [
        "savedmodel_zip",
        "mappings_path",
        "training_results_path",
        "classifier_spec_path",
    ]
    for k in expected:
        assert os.path.exists(artifacts[k])
