# Makefile for ClickHouse dev container

# Load variables from .env if present (CLICKHOUSE_USER, CLICKHOUSE_PASSWORD, CLICKHOUSE_DB, etc.)
-include .env

IMAGE_NAME      = clickhouse/clickhouse-server:latest
CONTAINER_NAME  = clickhouse_server_auto
LOCAL_DB_DIR   := $(shell pwd)/db_data

# Sensible defaults if .env is missing some values
CLICKHOUSE_USER      ?= default
CLICKHOUSE_PASSWORD  ?= defaultpw
CLICKHOUSE_DB        ?= trajectories

default: run

.PHONY: network run stop logs bash

## Ensure docker network exists
network:
	@docker network inspect clickhouse >/dev/null 2>&1 || \
		docker network create clickhouse

## Run ClickHouse with persistent storage and credentials from .env
run: network stop
	mkdir -p "$(LOCAL_DB_DIR)"
	docker run -d --name $(CONTAINER_NAME) \
		--network clickhouse \
		-p 9003:9000 -p 8123:8123 \
		-e CLICKHOUSE_USER=$(CLICKHOUSE_USER) \
		-e CLICKHOUSE_PASSWORD=$(CLICKHOUSE_PASSWORD) \
		-v "$(LOCAL_DB_DIR):/var/lib/clickhouse" \
		--restart=always \
		$(IMAGE_NAME)

## Stop and remove the container if it exists
stop down:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

## View container logs
logs:
	docker logs -f $(CONTAINER_NAME)

## Exec into the container with a bash shell
bash:
	docker exec -it $(CONTAINER_NAME) bash
