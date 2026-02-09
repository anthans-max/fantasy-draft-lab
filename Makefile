.PHONY: doctor acr-build acr-tags
TAG ?= v1
REPO ?= fantasy-draft-ui

doctor:
	@test -f Dockerfile || (echo "ERROR: Dockerfile not found in $(PWD)"; exit 1)
	@echo "PWD: $(PWD)"
	@ls -la Dockerfile requirements*.txt app.py 2>/dev/null || true

acr-build:
	az acr build --registry lotusacr31199 --image $(REPO):$(TAG) .

acr-tags:
	az acr repository show-tags --name lotusacr31199 --repository $(REPO) --orderby time_desc -o table
