-- Create databases for AI Stack components
CREATE DATABASE IF NOT EXISTS langfuse;
CREATE DATABASE IF NOT EXISTS n8n;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE langfuse TO trading;
GRANT ALL PRIVILEGES ON DATABASE n8n TO trading;
