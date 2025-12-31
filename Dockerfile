FROM postgres:16-bullseye

# 1. Install Build Dependencies
RUN apt-get update && apt-get install -y \
    postgresql-server-dev-16 \
    gcc \
    make \
    git \
    build-essential \
    libcurl4-openssl-dev \
    flex \
    bison \
    pkg-config \
    libxml2-dev

# 2. Install pgvector (v0.7.0 is stable)
RUN cd /tmp && \
    git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install

# 3. Install pgsql-http
RUN cd /tmp && \
    git clone https://github.com/pramsey/pgsql-http.git && \
    cd pgsql-http && \
    make && \
    make install

# 4. Install Apache AGE (Graph) for PG16
RUN cd /tmp && \
    git clone https://github.com/apache/age.git && \
    cd age && \
    # AGE versioning can be tricky, ensuring we are on a PG16 compatible branch/tag
    git checkout PG16 && \
    make && \
    make install

# 5. Cleanup to keep image small
RUN rm -rf /tmp/* && \
    apt-get purge -y --auto-remove \
    postgresql-server-dev-16 \
    gcc \
    make \
    git \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Run schema/init on first database startup.
COPY db/schema.sql /docker-entrypoint-initdb.d/init.sql
