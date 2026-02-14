.PHONY: build preprocess train predict shell

RUN := docker compose run --rm leafmine

build:
	docker compose build

preprocess:
	$(RUN) python scripts/prepare_data.py --config configs/preprocess.yaml

train:
	$(RUN) python scripts/train.py --config configs/train.yaml

predict:
	$(RUN) python scripts/predict.py --config configs/infer.yaml

shell:
	$(RUN) bash
