# MHAI Database Setup

This document outlines the PostgreSQL database setup for the Mental Health AI (MHAI) platform. The database schema includes user profiles, mental health assessments, and chat messages with vector embeddings for the RAG (Retrieval-Augmented Generation) system.

## Prerequisites

- PostgreSQL 15.0 or higher
- `pgvector` extension installed
- PostgreSQL UUID extension
- Supabase (if using Supabase as the backend)

## Database Schema Setup

Execute the following SQL commands in sequence to set up the database schema:

### 1. Enable Required Extensions

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Create User Profiles Table

```sql
CREATE TABLE user_profiles (
  id uuid REFERENCES auth.users ON DELETE CASCADE,
  full_name text NOT NULL,
  age integer CHECK (age > 0 AND age < 120),
  gender text,
  occupation text,
  emergency_contact text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  PRIMARY KEY (id)
);
```

### 3. Create Mental Health Assessments Table

```sql
CREATE TABLE mental_health_assessments (
  id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
  user_id uuid REFERENCES auth.users ON DELETE CASCADE,
  assessment_date timestamp with time zone DEFAULT now(),
  -- GAD-7 Assessment
  gad7_score integer CHECK (gad7_score BETWEEN 0 AND 21),
  gad7_severity text CHECK (gad7_severity IN ('Minimal', 'Mild', 'Moderate', 'Severe')),
  -- PHQ-9 Assessment
  phq9_score integer CHECK (phq9_score BETWEEN 0 AND 27),
  phq9_severity text CHECK (phq9_severity IN ('None-minimal', 'Mild', 'Moderate', 'Moderately Severe', 'Severe')),
  -- PTSD Assessment
  ptsd_positive boolean DEFAULT false,
  -- Suicide Risk Assessment
  has_suicidal_thoughts boolean DEFAULT false,
  -- Store all raw question responses
  raw_responses jsonb,
  created_at timestamp with time zone DEFAULT now()
);
```

### 4. Create Chat Messages Table

```sql
CREATE TABLE chat_messages (
  id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
  user_id uuid REFERENCES auth.users ON DELETE CASCADE,
  content text NOT NULL,
  is_bot boolean DEFAULT false,
  embedding vector(2048),
  created_at timestamp with time zone DEFAULT now()
);
```

### 5. Enable Real-time Updates

```sql
ALTER TABLE chat_messages REPLICA IDENTITY FULL;
CREATE PUBLICATION chat_messages_pub FOR TABLE chat_messages;
```

### 6. Configure Row Level Security

```sql
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE mental_health_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
```

### 7. Create Security Policies for User Profiles

```sql
CREATE POLICY "Users can read own profile"
  ON user_profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON user_profiles FOR UPDATE
  USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile"
  ON user_profiles FOR INSERT
  WITH CHECK (auth.uid() = id);
```

### 8. Create Security Policies for Mental Health Assessments

```sql
CREATE POLICY "Users can read own assessments"
  ON mental_health_assessments FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own assessments"
  ON mental_health_assessments FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own assessments"
  ON mental_health_assessments FOR UPDATE
  USING (auth.uid() = user_id);
```

### 9. Create Security Policies for Chat Messages

```sql
CREATE POLICY "Users can read own messages"
  ON chat_messages FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own messages"
  ON chat_messages FOR INSERT
  WITH CHECK (auth.uid() = user_id);
```

### 10. Create Message Matching Function for RAG

```sql
CREATE OR REPLACE FUNCTION match_messages(
  query_embedding vector(2048),
  match_threshold float,
  match_count int,
  user_id uuid
)
RETURNS TABLE (
  id uuid,
  content text,
  is_bot boolean,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    chat_messages.id,
    chat_messages.content,
    chat_messages.is_bot,
    1 - (chat_messages.embedding <=> query_embedding) AS similarity
  FROM chat_messages
  WHERE 1 - (chat_messages.embedding <=> query_embedding) > match_threshold
    AND chat_messages.user_id = match_messages.user_id
  ORDER BY chat_messages.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

### 11. Update Vector Dimensions and Create HNSW Index

```sql
-- Drop existing index if any
DROP INDEX IF EXISTS embedding_idx;

-- Modify table to support 2000-dimension vectors
ALTER TABLE chat_messages 
ALTER COLUMN embedding TYPE vector(2000);

-- Create HNSW index
CREATE INDEX embedding_idx ON chat_messages 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

> **Note**: The index parameters `m` and `ef_construction` control the trade-off between search speed and accuracy. `m` defines the maximum number of connections per layer (default is 16), and `ef_construction` defines the size of the dynamic candidate list during construction (default is 64).

### 12. Create Enhanced Chat Messages Table (v2)

```sql
CREATE TABLE chat_messages_v2 (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    content TEXT NOT NULL,
    reply_to UUID REFERENCES chat_messages_v2(id),
    role VARCHAR(20) CHECK (role IN ('user', 'bot', 'therapist')) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    embedding vector(2000)
);
```

### 13. Create Auto-update Timestamp Function

```sql
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE 'plpgsql';

-- Create trigger for updated_at
CREATE TRIGGER update_chat_messages_v2_updated_at
    BEFORE UPDATE ON chat_messages_v2
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

### 14. Configure Real-time for v2 and Row Level Security

```sql
ALTER TABLE chat_messages_v2 REPLICA IDENTITY FULL;
ALTER TABLE chat_messages_v2 ENABLE ROW LEVEL SECURITY;
```

### 15. Create Security Policies for Chat Messages v2

```sql
CREATE POLICY "Users can view their own messages"
    ON chat_messages_v2
    FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own messages"
    ON chat_messages_v2
    FOR INSERT
    WITH CHECK (auth.uid() = user_id AND role = 'user');

CREATE POLICY "Therapists can view all messages"
    ON chat_messages_v2
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM user_profiles 
            WHERE id = auth.uid() 
            AND role = 'therapist'
        )
    );

CREATE POLICY "Therapists can update messages"
    ON chat_messages_v2
    FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM user_profiles 
            WHERE id = auth.uid() 
            AND role = 'therapist'
        )
    );

CREATE POLICY "Bot can insert messages"
    ON chat_messages_v2
    FOR INSERT
    WITH CHECK (role = 'bot');
```

### 16. Create Updated Message Matching Function for v2

```sql
CREATE OR REPLACE FUNCTION match_messages(
    query_embedding vector(2000),
    match_threshold float,
    match_count int,
    user_id uuid
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        chat_messages_v2.id,
        chat_messages_v2.content,
        1 - (chat_messages_v2.embedding <=> query_embedding) as similarity
    FROM chat_messages_v2
    WHERE 1 - (chat_messages_v2.embedding <=> query_embedding) > match_threshold
    AND user_id = $4
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;
```

### 17. Update Chat Message Access Policies

```sql
-- Drop existing policies
DROP POLICY IF EXISTS "Users can read own messages" ON chat_messages;
DROP POLICY IF EXISTS "Users can read messages from their conversations" ON chat_messages;
DROP POLICY IF EXISTS "Users can read their conversations" ON chat_messages;
DROP POLICY IF EXISTS "Users can read messages" ON chat_messages;

-- Create a simple policy
CREATE POLICY "Users can read own messages"
  ON chat_messages FOR SELECT
  USING (auth.uid() = user_id);
```

## Commented-out Code (For Future Implementation)

The following policies and functions have been commented out in the current implementation but may be needed for future development:

```sql
-- Add therapist role to user_profiles
/*
ALTER TABLE user_profiles 
ADD COLUMN role VARCHAR(20) CHECK (role IN ('user', 'therapist')) DEFAULT 'user';
*/

-- Extended policies for user and therapist access
/*
CREATE POLICY "Therapists can read all messages"
  ON chat_messages FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM user_profiles
      WHERE user_profiles.id = auth.uid()
      AND user_profiles.role = 'therapist'
    )
    OR 
    -- Regular users can still see their conversations
    EXISTS (
      SELECT 1 
      FROM chat_messages AS cm 
      WHERE cm.user_id = auth.uid()
    )
  );

-- Create policy for therapists to update bot messages
CREATE POLICY "Therapists can update bot messages"
  ON chat_messages FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM user_profiles
      WHERE user_profiles.id = auth.uid()
      AND user_profiles.role = 'therapist'
    )
    AND is_bot = true  -- Only allow updating bot messages
  );

-- Function to check if user is a therapist
CREATE OR REPLACE FUNCTION is_therapist()
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM user_profiles
    WHERE user_profiles.id = auth.uid()
    AND user_profiles.role = 'therapist'
  );
END;
$$;
*/
```

## Production Security Considerations

For production environments, ensure the following:

1. All commented-out RLS policies should be properly implemented and tested
2. Database access should be restricted to only the necessary services
3. Sensitive information should be encrypted at rest 
4. Regular backups of the database should be configured
5. Monitor database performance and query execution plans, especially for vector operations
6. Index performance should be evaluated with production-scale data

## Vector Search Parameters Tuning

The HNSW index parameters can be adjusted for optimal performance:

- Increase `m` for better search quality at the cost of slower indexing and more memory
- Increase `ef_construction` for better index quality at the cost of slower indexing
- For production, consider experimenting with different parameter values and benchmark performance

## License

Copyright © 2025 Dandelion Fly. All rights reserved.

This is proprietary software. Unauthorized copying, modification, distribution, or use of this software, via any medium, is strictly prohibited without express written permission from Dandelion Fly.