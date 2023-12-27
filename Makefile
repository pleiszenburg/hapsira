DOCKER_BASE_IMAGE := "hapsira:dev"
DOCKER_CONTAINER_NAME := "hapsira-dev"

_clean_coverage:
	coverage erase

_clean_py:
	find src/ tests/ contrib/ -name '*.pyc' -exec rm -f {} +
	find src/ tests/ contrib/ -name '*.pyo' -exec rm -f {} +
	find src/ tests/ contrib/ -name '*~' -exec rm -f {} +
	find src/ tests/ contrib/ -name '__pycache__' -exec rm -fr {} +

_clean_release:
	-rm -r build/*
	-rm -r dist/*

clean:
	make _clean_release
	make _clean_coverage
	make _clean_py

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

test:
	DISPLAY= tox

.PHONY: docs docker image release upload
