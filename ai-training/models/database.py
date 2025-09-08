from supabase import create_client, Client
from config import settings
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.client: Client = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            logger.info("✅ Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            raise
    
    async def get_student(self, student_id: int):
        """Get student information by ID"""
        try:
            response = self.client.table("students").select("*").eq("id", student_id).execute()
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching student {student_id}: {e}")
            raise

    async def get_student_by_school_id(self, school_student_id: str):
        """Get student by school/student_id string column"""
        try:
            # Normalize input (trim whitespace)
            candidate = (school_student_id or "").strip()
            # Try exact match first
            response = self.client.table("students").select("*").eq("student_id", candidate).execute()
            if not response.data or len(response.data) == 0:
                # Try uppercase variant in case data is stored uppercased
                upper = candidate.upper()
                if upper and upper != candidate:
                    response = self.client.table("students").select("*").eq("student_id", upper).execute()
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching student by student_id {school_student_id}: {e}")
            raise
    
    async def create_trained_model(self, model_data: dict):
        """Create a new trained model record"""
        try:
            # Supabase Python v2 returns inserted rows by default (representation)
            response = self.client.table("trained_models").insert(model_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error creating trained model: {e}")
            raise
    
    async def get_trained_models(self, student_id: int = None):
        """Get trained models, optionally filtered by student"""
        try:
            query = self.client.table("trained_models").select("*")
            if student_id:
                query = query.eq("student_id", student_id)
            response = query.execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching trained models: {e}")
            raise
    
    async def update_model_status(self, model_id: int, status: str, accuracy: float = None):
        """Update model training status"""
        try:
            update_data = {"status": status}
            if accuracy is not None:
                update_data["accuracy"] = accuracy
            
            response = self.client.table("trained_models").update(update_data).eq("id", model_id).execute()
            # Some versions return None on update; swallow return
            return response.data[0] if getattr(response, 'data', None) else None
        except Exception as e:
            logger.error(f"Error updating model status: {e}")
            raise

    async def update_model_metadata(self, model_id: int, metadata: dict):
        """Update model metadata fields (e.g., prototype centroid/threshold)"""
        try:
            response = self.client.table("trained_models").update(metadata).eq("id", model_id).execute()
            return response.data[0] if getattr(response, 'data', None) else None
        except Exception as e:
            logger.error(f"Error updating model metadata: {e}")
            raise

    # Model Versioning Methods
    async def create_model_version(self, version_data: dict):
        """Create a new model version record."""
        try:
            response = self.client.table("model_versions").insert(version_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error creating model version: {e}")
            raise

    async def get_model_versions(self, model_id: int):
        """Get all versions for a model."""
        try:
            response = self.client.table("model_versions").select("*").eq("model_id", model_id).order("version", desc=False).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []

    async def get_model_version(self, version_id: int):
        """Get a specific model version."""
        try:
            response = self.client.table("model_versions").select("*, trained_models!inner(student_id)").eq("id", version_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting model version: {e}")
            return None

    async def get_active_model(self, student_id: int):
        """Get the active model version for a student."""
        try:
            response = self.client.table("model_versions").select("*, trained_models!inner(*)").eq("trained_models.student_id", student_id).eq("is_active", True).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting active model: {e}")
            return None

    async def activate_model_version(self, version_id: int):
        """Activate a model version."""
        try:
            response = self.client.table("model_versions").update({"is_active": True}).eq("id", version_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error activating model version: {e}")
            raise

    async def deactivate_other_versions(self, student_id: int, exclude_model_id: int = None):
        """Deactivate all other versions for a student."""
        try:
            query = self.client.table("model_versions").update({"is_active": False})
            query = query.eq("trained_models.student_id", student_id)
            if exclude_model_id:
                query = query.neq("model_id", exclude_model_id)
            response = query.execute()
            return response.data
        except Exception as e:
            logger.error(f"Error deactivating other versions: {e}")
            raise

    async def get_trained_model(self, model_id: int):
        """Get a trained model by ID."""
        try:
            response = self.client.table("trained_models").select("*").eq("id", model_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting trained model: {e}")
            return None

    # Student Signatures (S3-backed) -------------------------------------
    async def add_student_signature(self, student_id: int, label: str, s3_key: str, s3_url: str):
        """Insert a student signature record referencing S3 storage."""
        try:
            response = self.client.table("student_signatures").insert({
                "student_id": student_id,
                "label": label,
                "s3_key": s3_key,
                "s3_url": s3_url,
            }).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error adding student signature: {e}")
            raise

    async def list_student_signatures(self, student_id: int):
        """List all signatures for a given student."""
        try:
            response = self.client.table("student_signatures").select("*").eq("student_id", student_id).order("created_at", desc=False).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error listing student signatures: {e}")
            return []

    async def list_all_signatures(self):
        """List all signatures across students for global training manifest."""
        try:
            response = self.client.table("student_signatures").select("student_id,label,s3_key,s3_url,created_at").order("student_id", desc=False).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error listing all signatures: {e}")
            return []

    async def find_signature_by_hash(self, content_hash: str):
        try:
            response = self.client.table("student_signatures").select("*").eq("content_hash", content_hash).limit(1).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error finding signature by hash: {e}")
            return None

    async def add_signature_with_hash(self, student_id: int, label: str, s3_key: str, s3_url: str, content_hash: str):
        try:
            response = self.client.table("student_signatures").insert({
                "student_id": student_id,
                "label": label,
                "s3_key": s3_key,
                "s3_url": s3_url,
                "content_hash": content_hash,
            }).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error adding signature with hash: {e}")
            raise

    async def delete_signature(self, record_id: int):
        try:
            response = self.client.table("student_signatures").delete().eq("id", record_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error deleting signature: {e}")
            return False

    async def list_students_with_images(self):
        try:
            # Select distinct students with aggregated signatures
            response = self.client.rpc("list_students_with_images").execute()
            return response.data or []
        except Exception:
            # Fallback if RPC not present
            try:
                rows = self.client.table("student_signatures").select("student_id,label,s3_url").execute().data or []
                by = {}
                for r in rows:
                    sid = r["student_id"]
                    by.setdefault(sid, []).append(r)
                out = []
                for sid, items in by.items():
                    out.append({"student_id": sid, "signatures": items})
                return out
            except Exception as e2:
                logger.error(f"Error listing students with images: {e2}")
                return []

    # Global Models (separate table) -------------------------------------
    async def create_global_model(self, model_data: dict):
        """Create a new global trained model record."""
        try:
            response = self.client.table("global_trained_models").insert(model_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error creating global model: {e}")
            raise

    async def get_global_models(self, limit: int = None):
        """Get all global trained models, optionally limited."""
        try:
            query = self.client.table("global_trained_models").select("*").order("created_at", desc=True)
            if limit:
                query = query.limit(limit)
            response = query.execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting global models: {e}")
            return []

    async def get_global_model(self, model_id: int):
        """Get a specific global model by ID."""
        try:
            response = self.client.table("global_trained_models").select("*").eq("id", model_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting global model: {e}")
            return None

    async def get_latest_global_model(self):
        """Get the most recent global model."""
        try:
            response = self.client.table("global_trained_models").select("*").order("created_at", desc=True).limit(1).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting latest global model: {e}")
            return None

    async def get_latest_ai_model(self):
        """Return the most recent AI (embedding-based) individual model, if any."""
        try:
            # Fetch the latest N trained models and filter on the client side by training_metrics.model_type
            response = self.client.table("trained_models").select("*").order("created_at", desc=True).limit(50).execute()
            rows = response.data or []
            for r in rows:
                metrics = r.get("training_metrics", {}) or {}
                model_type = str(metrics.get("model_type", ""))
                if model_type in ("ai_signature_verification", "ai_signature_verification_gpu"):
                    return r
            return None
        except Exception as e:
            logger.error(f"Error getting latest AI model: {e}")
            return None

    async def update_global_model_status(self, model_id: int, status: str, accuracy: float = None):
        """Update global model training status."""
        try:
            update_data = {"status": status}
            if accuracy is not None:
                update_data["accuracy"] = accuracy
            
            response = self.client.table("global_trained_models").update(update_data).eq("id", model_id).execute()
            return response.data[0] if getattr(response, 'data', None) else None
        except Exception as e:
            logger.error(f"Error updating global model status: {e}")
            raise

    async def delete_global_model(self, model_id: int):
        """Delete a global model record."""
        try:
            response = self.client.table("global_trained_models").delete().eq("id", model_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error deleting global model: {e}")
            return False

    # A/B Testing Methods
    async def create_ab_test(self, ab_test_data: dict):
        """Create an A/B test."""
        try:
            response = self.client.table("model_ab_tests").insert(ab_test_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error creating A/B test: {e}")
            raise

    async def get_active_ab_tests(self, student_id: int):
        """Get active A/B tests for a student."""
        try:
            response = self.client.table("model_ab_tests").select("*").eq("student_id", student_id).eq("is_active", True).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting active A/B tests: {e}")
            return []

    async def get_ab_test_results(self, ab_test_id: int):
        """Get A/B test results."""
        try:
            response = self.client.table("verification_results").select("*, trained_models!inner(*)").eq("ab_test_id", ab_test_id).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting A/B test results: {e}")
            return []

    # Verification Results
    async def create_verification_result(self, result_data: dict):
        """Create a verification result record."""
        try:
            response = self.client.table("verification_results").insert(result_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error creating verification result: {e}")
            raise

    # Audit Trail
    async def create_audit_log(self, audit_data: dict):
        """Create an audit log entry."""
        try:
            response = self.client.table("model_audit_log").insert(audit_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error creating audit log: {e}")
            raise

    async def get_model_audit_trail(self, model_id: int):
        """Get audit trail for a model."""
        try:
            response = self.client.table("model_audit_log").select("*").eq("model_id", model_id).order("performed_at", desc=True).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting audit trail: {e}")
            return []

    async def get_student_model_history(self, student_id: int):
        """Get complete model history for a student."""
        try:
            response = self.client.table("trained_models").select("*, model_versions(*)").eq("student_id", student_id).order("training_date", desc=True).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting student model history: {e}")
            return []

# Global database manager instance
db_manager = DatabaseManager()
