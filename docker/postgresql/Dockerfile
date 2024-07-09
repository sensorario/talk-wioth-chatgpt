FROM postgres:latest

RUN apt-get update && \
    apt-get install -y git build-essential postgresql-server-dev-all

WORKDIR /docker/postgres
COPY ./docker/postgresql/pg_hba.conf /var/lib/postgresql/data/pg_hba.conf
COPY ./docker/postgresql/postgresql.conf /var/lib/postgresql/data/postgresql.conf
COPY ./docker/postgresql/init.sql /docker-entrypoint-initdb.d/init.sql

WORKDIR /tmp
RUN git clone --branch v0.4.4 https://github.com/pgvector/pgvector.git
WORKDIR /tmp/pgvector
RUN make
RUN make install



ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=testpassword
ENV POSTGRES_DB=vectordb
ENV PGUSER=postgres

EXPOSE 5432

# CMD ["postgres", "-c", "config_file=/var/lib/postgresql/data/pg_hba.conf"]
CMD ["-c", "config_file=/var/lib/postgresql/data/postgresql.conf"]


