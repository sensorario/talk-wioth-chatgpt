stop:
	docker compose stop postgres

rm:
	rm -rf postgres_data/**
	docker compose rm postgres --force
	docker volume ls -qf dangling=true | xargs -r docker volume rm

up:
	docker compose up
	
rebuild: stop rm up

run:
	python3 solobert.py

bash:
	docker exec -it mio-container-postgresql bash