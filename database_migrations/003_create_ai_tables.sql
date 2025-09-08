-- Migration: Create AI Signature Verification Tables
-- This migration creates all necessary tables for the AI signature verification system

-- Create student_signatures table for S3-backed signature storage
CREATE TABLE IF NOT EXISTS public.student_signatures (
    id BIGSERIAL PRIMARY KEY,
    student_id BIGINT NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
    label TEXT NOT NULL CHECK (label IN ('genuine', 'forged')),
    s3_key TEXT NOT NULL,
    s3_url TEXT NOT NULL,
    content_hash TEXT, -- For duplicate detection
    file_size INTEGER,
    file_type TEXT,
    quality_score FLOAT CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create trained_models table for individual student models
CREATE TABLE IF NOT EXISTS public.trained_models (
    id BIGSERIAL PRIMARY KEY,
    student_id BIGINT NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
    model_path TEXT NOT NULL, -- S3 URL to the model file
    embedding_model_path TEXT, -- S3 URL to the embedding model
    s3_key TEXT, -- S3 object key for direct access
    model_uuid TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'training' CHECK (status IN ('training', 'completed', 'failed')),
    sample_count INTEGER NOT NULL DEFAULT 0,
    genuine_count INTEGER NOT NULL DEFAULT 0,
    forged_count INTEGER NOT NULL DEFAULT 0,
    training_date TIMESTAMPTZ DEFAULT NOW(),
    accuracy NUMERIC(5,4),
    training_metrics JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    parent_model_id BIGINT REFERENCES public.trained_models(id),
    is_active BOOLEAN DEFAULT true,
    version_notes TEXT,
    performance_metrics JSONB,
    far NUMERIC(5,4), -- False Acceptance Rate
    frr NUMERIC(5,4)  -- False Rejection Rate
);

-- Create model_versions table for versioning
CREATE TABLE IF NOT EXISTS public.model_versions (
    id BIGSERIAL PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES public.trained_models(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    model_path TEXT NOT NULL,
    embedding_model_path TEXT,
    s3_key TEXT,
    model_uuid TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'training' CHECK (status IN ('training', 'completed', 'failed')),
    accuracy NUMERIC(5,4),
    training_metrics JSONB,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version_notes TEXT,
    performance_metrics JSONB,
    UNIQUE(model_id, version)
);

-- Create model_audit_log table for tracking changes
CREATE TABLE IF NOT EXISTS public.model_audit_log (
    id BIGSERIAL PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES public.trained_models(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    performed_by TEXT,
    performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    old_values JSONB,
    new_values JSONB,
    notes TEXT
);

-- Create A/B testing table
CREATE TABLE IF NOT EXISTS public.model_ab_tests (
    id BIGSERIAL PRIMARY KEY,
    student_id BIGINT NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
    model_a_id BIGINT NOT NULL REFERENCES public.trained_models(id) ON DELETE CASCADE,
    model_b_id BIGINT NOT NULL REFERENCES public.trained_models(id) ON DELETE CASCADE,
    test_name TEXT NOT NULL,
    description TEXT,
    start_date TIMESTAMPTZ DEFAULT NOW(),
    end_date TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true,
    traffic_split DECIMAL(3,2) DEFAULT 0.5, -- 0.5 = 50/50 split
    results JSONB,
    created_by TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create verification results table for A/B testing
CREATE TABLE IF NOT EXISTS public.verification_results (
    id BIGSERIAL PRIMARY KEY,
    student_id BIGINT NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
    model_id BIGINT NOT NULL REFERENCES public.trained_models(id) ON DELETE CASCADE,
    ab_test_id BIGINT REFERENCES public.model_ab_tests(id) ON DELETE SET NULL,
    test_signature_path TEXT,
    verification_result JSONB, -- Store full verification response
    processing_time_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_student_signatures_student_id ON public.student_signatures(student_id);
CREATE INDEX IF NOT EXISTS idx_student_signatures_label ON public.student_signatures(label);
CREATE INDEX IF NOT EXISTS idx_student_signatures_content_hash ON public.student_signatures(content_hash);
CREATE INDEX IF NOT EXISTS idx_student_signatures_created_at ON public.student_signatures(created_at);

CREATE INDEX IF NOT EXISTS idx_trained_models_student_id ON public.trained_models(student_id);
CREATE INDEX IF NOT EXISTS idx_trained_models_status ON public.trained_models(status);
CREATE INDEX IF NOT EXISTS idx_trained_models_active ON public.trained_models(student_id, is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_trained_models_created_at ON public.trained_models(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trained_models_model_uuid ON public.trained_models(model_uuid);

CREATE INDEX IF NOT EXISTS idx_model_versions_model_id ON public.model_versions(model_id);
CREATE INDEX IF NOT EXISTS idx_model_versions_active ON public.model_versions(model_id, is_active) WHERE is_active = true;

CREATE INDEX IF NOT EXISTS idx_model_audit_log_model_id ON public.model_audit_log(model_id);
CREATE INDEX IF NOT EXISTS idx_model_ab_tests_student_id ON public.model_ab_tests(student_id);
CREATE INDEX IF NOT EXISTS idx_verification_results_student_id ON public.verification_results(student_id);
CREATE INDEX IF NOT EXISTS idx_verification_results_model_id ON public.verification_results(model_id);
CREATE INDEX IF NOT EXISTS idx_verification_results_ab_test_id ON public.verification_results(ab_test_id);

-- Create trigger functions for updating timestamps
CREATE OR REPLACE FUNCTION update_student_signatures_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_trained_models_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at columns
CREATE TRIGGER update_student_signatures_updated_at
    BEFORE UPDATE ON public.student_signatures
    FOR EACH ROW EXECUTE FUNCTION update_student_signatures_updated_at();

CREATE TRIGGER update_trained_models_updated_at
    BEFORE UPDATE ON public.trained_models
    FOR EACH ROW EXECUTE FUNCTION update_trained_models_updated_at();

-- Create function to get students with signature images
CREATE OR REPLACE FUNCTION list_students_with_images()
RETURNS TABLE (
    student_id BIGINT,
    signatures JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ss.student_id,
        jsonb_agg(
            jsonb_build_object(
                'id', ss.id,
                'label', ss.label,
                's3_url', ss.s3_url,
                's3_key', ss.s3_key,
                'quality_score', ss.quality_score,
                'created_at', ss.created_at
            )
        ) as signatures
    FROM public.student_signatures ss
    GROUP BY ss.student_id
    ORDER BY ss.student_id;
END;
$$ LANGUAGE plpgsql;

-- Add comments for documentation
COMMENT ON TABLE public.student_signatures IS 'Stores metadata for student signature images stored in S3';
COMMENT ON TABLE public.trained_models IS 'Stores metadata for individual student signature verification models';
COMMENT ON TABLE public.model_versions IS 'Tracks different versions of trained models for rollback capability';
COMMENT ON TABLE public.model_audit_log IS 'Audit trail for model changes and operations';
COMMENT ON TABLE public.model_ab_tests IS 'A/B testing configuration for comparing model performance';
COMMENT ON TABLE public.verification_results IS 'Stores verification results for analysis and A/B testing';

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_app_user;
