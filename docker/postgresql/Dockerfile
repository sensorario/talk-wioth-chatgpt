FROM postgres:latest

WORKDIR /docker/postgres
COPY ./docker/postgresql/pg_hba.conf /var/lib/postgresql/data/pg_hba.conf
COPY ./docker/postgresql/postgresql.conf /var/lib/postgresql/data/postgresql.conf

ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=testpassword
ENV POSTGRES_DB=vectordb
ENV PGUSER=postgres

EXPOSE 5432

# CMD ["postgres", "-c", "config_file=/var/lib/postgresql/data/pg_hba.conf"]
CMD ["-c", "config_file=/var/lib/postgresql/data/postgresql.conf"]


