FROM postgres:latest

# COPY /Volumes/DEVELOPMENT/localhost/dddemo3/docker/postgresql/pg_hba.conf /var/lib/postgresql/data/pg_hba.conf
COPY pg_hba.conf /var/lib/postgresql/data/pg_hba.conf
COPY postgresql.conf /var/lib/postgresql/data/postgresql.conf

ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=testpassword
ENV POSTGRES_DB=vectordb

EXPOSE 5432

# CMD ["postgres", "-c", "config_file=/var/lib/postgresql/data/pg_hba.conf"]
CMD ["postgres", "-c", "config_file=/var/lib/postgresql/data/postgresql.conf"]


