#!/bin/bash
set -e

READER_PASS="${NLQ_READER_PASSWORD:-secret_nlq_reader}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    DO \$\$
    BEGIN
      IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ai_nlq_reader') THEN
        CREATE ROLE ai_nlq_reader LOGIN PASSWORD '$READER_PASS';
      END IF;
    END
    \$\$;

    GRANT CONNECT ON DATABASE $POSTGRES_DB TO ai_nlq_reader;
    GRANT USAGE ON SCHEMA public TO ai_nlq_reader;

    -- Cấp quyền SELECT trên các bảng dữ liệu bán hàng
    GRANT SELECT ON categories, suppliers, products, warehouses, orders,
      order_items, inventory, stock_movements TO ai_nlq_reader;

    -- Giới hạn quyền trên customers: chỉ cho đọc các cột kinh doanh, tuyệt đối không lộ email và phone
    GRANT SELECT (customer_id, city, member_tier, joined_on)
      ON customers TO ai_nlq_reader;

    -- Đảm bảo không cấp quyền đọc bảng rag_chunks cho tài khoản NLQ
    REVOKE ALL ON TABLE rag_chunks FROM ai_nlq_reader;
EOSQL
