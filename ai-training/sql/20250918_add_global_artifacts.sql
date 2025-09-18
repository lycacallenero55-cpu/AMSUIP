-- Add artifact columns to global_trained_models to support verification loading
ALTER TABLE public.global_trained_models
  ADD COLUMN IF NOT EXISTS mappings_path TEXT,
  ADD COLUMN IF NOT EXISTS embedding_spec_path TEXT,
  ADD COLUMN IF NOT EXISTS centroids_path TEXT;

-- Optional: ensure training_metrics exists as JSONB for richer logs
ALTER TABLE public.global_trained_models
  ADD COLUMN IF NOT EXISTS training_metrics JSONB;
