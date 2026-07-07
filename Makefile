.PHONY: help install train test run docker-build docker-run docker-stop

help:
	@echo ""
	@echo "  Network Anomaly Detection API"
	@echo "  ──────────────────────────────"
	@echo "  make install       Installe les dépendances (dev)"
	@echo "  make train         Entraîne et sauvegarde le modèle dummy"
	@echo "  make test          Lance les tests unitaires"
	@echo "  make run           Démarre l'API en local (dev)"
	@echo "  make docker-build  Build l'image Docker"
	@echo "  make docker-run    Lance via docker-compose"
	@echo "  make docker-stop   Arrête les containers"
	@echo ""

install:
	pip install -r requirements-dev.txt

train:
	@mkdir -p model
	python scripts/train_dummy_model.py

test:
	pytest tests/ -v --tb=short

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker compose build

docker-run:
	docker compose up -d
	@echo " API disponible sur http://localhost:8000"
	@echo " Docs Swagger : http://localhost:8000/docs"

docker-stop:
	docker compose down

logs:
	docker compose logs -f api
