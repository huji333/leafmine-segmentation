.PHONY: build preprocess remove_processed reprocess train predict shell

RUN := docker compose run --rm leafmine

build:
	DOCKER_BUILDKIT=1 docker compose build

preprocess:
	$(RUN) python scripts/prepare_data.py --config configs/preprocess.yaml

remove_processed:
	rm -rf data/processed data/splits

reprocess: remove_processed preprocess

train:
	$(RUN) python scripts/train.py --config configs/train.yaml

predict:
	$(RUN) python scripts/predict.py --config configs/infer.yaml

shell:
	$(RUN) bash
