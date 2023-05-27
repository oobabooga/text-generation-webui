(test -f ./Dockerfile && rm ./Dockerfile); \
(test -f ./.env && rm ./.env); \
(test -f ./.dockerignore && rm ./.dockerignore); \
(test -f ./docker-compose.yml && rm ./docker-compose.yml); \
ln -s docker/{Dockerfile,docker-compose.yml,.dockerignore,.env} . && docker compose up
