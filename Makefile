DOCKER_BASE_IMAGE := "hapsira:dev"
DOCKER_CONTAINER_NAME := "hapsira-dev"

docs:
	tox -e docs

image: Dockerfile pyproject.toml
	docker build \
	-t hapsira:dev \
	.

docker:
	docker run \
	-it \
	--rm \
	--name ${DOCKER_CONTAINER_NAME} \
	--volume $(shell pwd):/code \
	--user $(shell id -u):$(shell id -g) \
	${DOCKER_BASE_IMAGE} \
		bash

release:
	make clean
	flit build
	gpg --detach-sign -a dist/hapsira*.whl
	gpg --detach-sign -a dist/hapsira*.tar.gz

upload:
	for filename in $$(ls dist/*.tar.gz dist/*.whl) ; do \
		twine upload $$filename $$filename.asc ; \
	done

.PHONY: docs docker image release upload
