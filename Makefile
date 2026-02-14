.PHONY: build preprocess train predict shell

IMAGE_NAME := leafmine-segmentation
RUN := docker compose run --rm leafmine

build:
	docker compose build

preprocess:
	$(RUN) uv run python scripts/prepare_data.py --config configs/preprocess.yaml

train:
	$(RUN) uv run python scripts/train.py --config configs/train.yaml

predict:
	$(RUN) uv run python scripts/predict.py --config configs/infer.yaml

shell:
	$(RUN) bash
