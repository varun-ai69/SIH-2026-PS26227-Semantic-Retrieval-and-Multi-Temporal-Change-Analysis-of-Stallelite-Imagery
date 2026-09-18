-- =============================================================================
-- Migration 003: Analyst Workflow & Provenance Tables
-- =============================================================================
-- 1. analyst_decisions  : Immutable decision audit trail for Change Analysis
-- 2. search_feedback    : Relevance feedback audit trail for Semantic Retrieval
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. ANALYST_DECISIONS: Immutable append-only audit trail of analyst change decisions
CREATE TABLE IF NOT EXISTS analyst_decisions (
    decision_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id          TEXT NOT NULL REFERENCES change_runs(run_id) ON DELETE CASCADE,
    candidate_id    TEXT NOT NULL,
    analyst_id      VARCHAR(100) NOT NULL DEFAULT 'ANALYST-DEF-01',
    decision        VARCHAR(30) NOT NULL, -- 'confirmed', 'rejected', 'unsure'
    note            TEXT DEFAULT '',
    tags            JSONB DEFAULT '[]'::jsonb,
    decided_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analyst_decisions_run_cand
    ON analyst_decisions (run_id, candidate_id);
CREATE INDEX IF NOT EXISTS idx_analyst_decisions_decided_at
    ON analyst_decisions (decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyst_decisions_analyst
    ON analyst_decisions (analyst_id);
CREATE INDEX IF NOT EXISTS idx_analyst_decisions_decision
    ON analyst_decisions (decision);

-- 2. SEARCH_FEEDBACK: Relevance feedback for Semantic Retrieval (Rocchio input & curation)
CREATE TABLE IF NOT EXISTS search_feedback (
    feedback_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analyst_id      VARCHAR(100) NOT NULL DEFAULT 'ANALYST-DEF-01',
    query_text      TEXT NOT NULL,
    tile_id         TEXT REFERENCES tiles(tile_id) ON DELETE CASCADE,
    relevant        BOOLEAN NOT NULL, -- true = Relevant / Hit, false = Irrelevant / False Alarm
    relevance_score FLOAT DEFAULT 1.0, -- 1.0 = positive hit, 0.0 = negative
    tag             VARCHAR(100) DEFAULT '',
    note            TEXT DEFAULT '',
    recorded_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_feedback_query
    ON search_feedback (query_text, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_search_feedback_tile
    ON search_feedback (tile_id);
CREATE INDEX IF NOT EXISTS idx_search_feedback_analyst
    ON search_feedback (analyst_id);
CREATE INDEX IF NOT EXISTS idx_search_feedback_recorded
    ON search_feedback (recorded_at DESC);
