from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List

from models.database import db_manager
from utils.s3_storage import upload_bytes, create_presigned_post, create_presigned_get, delete_key, object_exists, count_student_signatures
from config import settings


router = APIRouter()

def _derive_s3_key_from_url(url: str) -> str | None:
    if not url:
        return None
    base = url.split('?', 1)[0]
    if "amazonaws.com" not in base:
        return None
    try:
        parts = base.split(".amazonaws.com/")
        if len(parts) == 2:
            return parts[1] or None
        return base.split("/", 3)[-1] or None
    except Exception:
        return None

@router.post("/signature")
async def upload_signature(
    student_id: int = Form(...),
    label: str = Form(...),  # 'genuine' | 'forged'
    file: UploadFile = File(...),
):
    if label not in ("genuine", "forged"):
        raise HTTPException(status_code=400, detail="label must be 'genuine' or 'forged'")
    data = await file.read()
    try:
        key, url = upload_bytes(student_id, label, file.filename or "signature.png", data, file.content_type)
        # Always store the signature - no duplicate prevention
        record = await db_manager.add_student_signature(student_id, label, key, url)
        return {"success": True, "record": record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/list")
async def list_signatures(student_id: int):
    try:
        rows = await db_manager.list_student_signatures(student_id)
        filtered = []
        for r in rows:
            key = r.get("s3_key") or _derive_s3_key_from_url(r.get("s3_url", ""))  # type: ignore[attr-defined]
            if key and object_exists(key):
                if settings.S3_USE_PRESIGNED_GET:
                    r["s3_url"] = create_presigned_get(key)  # type: ignore[index]
                filtered.append(r)
        return {"signatures": filtered}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


@router.post("/presign")
async def presign_upload(
    student_id: int = Form(...),
    label: str = Form(...),
    filename: str = Form(...),
    content_type: Optional[str] = Form(None),
):
    if label not in ("genuine", "forged"):
        raise HTTPException(status_code=400, detail="label must be 'genuine' or 'forged'")
    try:
        post = create_presigned_post(student_id, label, filename, content_type)
        return {"success": True, **post}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Presign failed: {str(e)}")


@router.get("/manifest")
async def generate_manifest():
    try:
        rows = await db_manager.list_all_signatures()
        return {"items": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manifest failed: {str(e)}")


@router.delete("/signature/{record_id}")
async def delete_signature(record_id: int, s3_key: Optional[str] = None):
    try:
        # if s3_key not provided, fetch it
        key = s3_key
        if not key:
            # quick fetch by id
            # Supabase select
            # We reuse list_all_signatures then filter to minimize new DB call surface
            # Better: add a dedicated DB method to get by id (omitted for brevity)
            from supabase import APIError  # type: ignore
            try:
                # direct fetch
                resp = db_manager.client.table("student_signatures").select("s3_key").eq("id", record_id).execute()
                if resp.data:
                    key = resp.data[0]["s3_key"]
            except Exception:
                pass
        if key:
            delete_key(key)
        ok = await db_manager.delete_signature(record_id)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to delete record")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.get("/students-with-images")
async def students_with_images(summary: bool = False):
    try:
        if summary:
            # Trust S3 as the source of truth: count objects under {student_id}/genuine|forged
            summarized = []
            try:
                # List candidate student IDs from DB quickly, then verify counts from S3
                resp = db_manager.client.table("student_signatures").select("student_id").execute()
                rows = resp.data or []
                seen = set()
                for r in rows:
                    sid = r.get("student_id")
                    if sid is None or sid in seen:
                        continue
                    seen.add(sid)
                    g, f = count_student_signatures(int(sid))
                    if (g + f) > 0:
                        summarized.append({
                            "student_id": int(sid),
                            "genuine_count": int(g),
                            "forged_count": int(f),
                        })
            except Exception:
                summarized = []
            return {"items": summarized}
        # Non-summary: original detailed validation path
        items = await db_manager.list_students_with_images()
        for it in items:
            valid = []
            for s in it.get("signatures", []):
                key = _derive_s3_key_from_url(s.get("s3_url", "")) or s.get("s3_key")
                if key and object_exists(key):
                    if settings.S3_USE_PRESIGNED_GET:
                        s["s3_url"] = create_presigned_get(key)
                    valid.append(s)
            it["signatures"] = valid
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List students failed: {str(e)}")


