import io
import os
import json
import shutil
import tempfile
import zipfile
from datetime import datetime
from typing import Dict, Tuple
from tensorflow import keras


def save_savedmodel(model: keras.Model, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    model.save(out_dir, save_format="tf")
    return out_dir


def zip_directory(src_dir: str, zip_path: str) -> str:
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, start=src_dir)
                zf.write(full, arcname=rel)
    return zip_path


def write_json(path: str, obj: Dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    return path


def build_classifier_spec(num_classes: int, image_size: int, backbone: str = "MobileNetV2") -> Dict:
    return {
        "type": "classification",
        "backbone": backbone,
        "image_size": image_size,
        "num_classes": num_classes,
        "created_at": datetime.utcnow().isoformat(),
    }


def package_global_classifier_artifacts(
    model_uuid: str,
    base_dir: str,
    classifier: keras.Model,
    mappings_id_first: Dict,
    centroids: Dict[int, list] | None,
    training_results: Dict,
    classifier_spec: Dict,
) -> Dict[str, str]:
    out_dir = os.path.join(base_dir, model_uuid)
    os.makedirs(out_dir, exist_ok=True)
    # SavedModel directory and zip
    savedmodel_dir = os.path.join(out_dir, f"{model_uuid}_classification.tf")
    save_savedmodel(classifier, savedmodel_dir)
    zip_path = os.path.join(out_dir, f"{model_uuid}_classification_savedmodel.zip")
    zip_directory(savedmodel_dir, zip_path)
    # Write JSON artifacts
    mappings_path = os.path.join(out_dir, f"{model_uuid}_mappings.json")
    write_json(mappings_path, mappings_id_first)
    if centroids is not None:
        centroids_path = os.path.join(out_dir, f"{model_uuid}_centroids.json")
        write_json(centroids_path, {str(k): v for k, v in centroids.items()})
    else:
        centroids_path = None
    results_path = os.path.join(out_dir, f"{model_uuid}_training_results.json")
    write_json(results_path, training_results)
    spec_path = os.path.join(out_dir, f"{model_uuid}_classifier_spec.json")
    write_json(spec_path, classifier_spec)
    return {
        "savedmodel_dir": savedmodel_dir,
        "savedmodel_zip": zip_path,
        "mappings_path": mappings_path,
        "centroids_path": centroids_path or "",
        "training_results_path": results_path,
        "classifier_spec_path": spec_path,
    }

