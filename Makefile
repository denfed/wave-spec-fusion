UID ?= $(shell id -u) # to set UID of container user to match host
GID ?= $(shell id -g) # to set GID of container user to match host
USER_PASSWORD ?= password # container user password (for sudo)
PORT?=6066 # tensorboard port

# Build without cache
.PHONY: build-nocache
build-nocache:
	docker-compose stop
	docker-compose build --no-cache --build-arg UID=$(UID) --build-arg GID=$(GID) --build-arg USER_PASSWORD=$(USER_PASSWORD) dcase_2021

# Build with cache
.PHONY: build
build:
	docker-compose stop
	docker-compose build --build-arg UID=$(UID) --build-arg GID=$(GID) --build-arg USER_PASSWORD=$(USER_PASSWORD) dcase_2021

# Start the container and the Jupyterlab environment
.PHONY: run
run:
	docker-compose stop
	docker-compose up -d dcase_2021

# Create terminal inside container
.PHONY: terminal
terminal:
	docker-compose run dcase_2021 bash

# Start tensorboard environment
.PHONY: tb
tb: 
	tensorboard --logdir=saved/log/ --port=$(PORT) --bind_all --max_reload_threads 4

# Start the wandb environment at port 8080
.PHONY: wandb
wandb:
	docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local

# Stop the wandb environment
.PHONY: wandb-stop
wandb-stop:
	docker stop wandb-local

# Upgrade the wandb container
.PHONY: wandb-upgrade
wandb-upgrade:
	docker pull wandb/local
