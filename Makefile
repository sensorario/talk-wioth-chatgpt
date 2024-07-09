stop:
	docker compose down
	docker compose stop postgres
	docker rmi dddemo3-postgres

rm:
	rm -rf pgdata/**
	docker compose rm postgres --force
	docker volume ls -qf dangling=true | xargs -r docker volume rm

up:
	docker compose up
	
rebuild: stop rm up

run:
	python3 solobert.py

bash:
	docker exec -it mio-container-postgresql bash