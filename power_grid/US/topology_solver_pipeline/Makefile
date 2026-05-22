# =============================================================================
# Makefile — topology_solver_pipeline (Docker + Local)
# =============================================================================
#
# Runs all pipeline stages either inside Docker or locally with a native
# Julia installation. Every Docker target has a local-* counterpart.
#
# Docker quick start:
#   make build
#   make run STATE=alabama HOUR=16h DATA_DIR=/path/to/GridSFM_US_power_grid
#
# Local quick start:
#   make local-install          # one-time: instantiate Julia packages
#   make local-run STATE=alabama HOUR=16h DATA_DIR=/path/to/GridSFM_US_power_grid
#
# See README.md for detailed usage.
# =============================================================================

# ── Configuration (override on command line) ─────────────────────────────────

# Docker image name
IMAGE         ?= topo_solver_pipe

# Path to the GridSFM US power grid data directory (contains 04h/ and 16h/ subdirs)
# Override: make run DATA_DIR=/path/to/GridSFM_US_power_grid
DATA_DIR      ?=

# Output directory on host
OUTPUT_DIR    ?= $(CURDIR)/output

# State and hour for pipeline runs
STATE         ?= alabama
HOUR          ?= 16h

# Number of perturbed scenarios per mode (5 modes total)
N_PER_MODE    ?= 1

# Number of Julia worker processes for scenario generation
N_WORKERS     ?= 2

# CPU core range for taskset pinning in bulk scenario generation (e.g. 0-7).
# Leave empty to auto-detect at runtime as 0-(N-1), where N is the number of
# logical CPUs (nproc on Linux, sysctl hw.logicalcpu on macOS). Note: taskset
# is Linux-only; on macOS this value is informational and pinning is skipped.
CPU_RANGE     ?=

# Grid list file for bulk generation (one "<solvable.json> <n_per_mode>" per line)
GRID_LIST     ?=

# For verify-one: path to a specific .pyg.json file (relative to OUTPUT_DIR)
PYG_FILE      ?=

# ── Derived paths ────────────────────────────────────────────────────────────

# topology_solver_pipeline root (this directory)
PIPE_DIR      := $(CURDIR)

# Julia project flag for local execution
JULIA_PROJECT := --project=$(PIPE_DIR)

# Guard: require DATA_DIR for targets that need input data
define check_data_dir
$(if $(DATA_DIR),,$(error DATA_DIR is not set. Set it to the path of your GridSFM_US_power_grid directory, e.g. make $@ DATA_DIR=/path/to/GridSFM_US_power_grid))
endef

# Docker run with both /data and /output mounts (for targets needing input data)
DOCKER_RUN    = docker run --rm \
                  -v $(DATA_DIR):/data:ro \
                  -v $(OUTPUT_DIR):/output \
                  $(IMAGE)

# Docker run with /output only (for targets that operate on pipeline outputs)
DOCKER_RUN_LOCAL = docker run --rm \
                     -v $(OUTPUT_DIR):/output \
                     $(IMAGE)

# =============================================================================
#  DOCKER TARGETS
# =============================================================================

# ── Build ────────────────────────────────────────────────────────────────────

.PHONY: build
build: ## [docker] Build the Docker image
	cd $(CURDIR) && docker build -t $(IMAGE) -f docker/Dockerfile .

.PHONY: rebuild
rebuild: ## [docker] Rebuild the Docker image from scratch (no cache)
	cd $(CURDIR) && docker build --no-cache -t $(IMAGE) -f docker/Dockerfile .

# ── Full Pipeline ────────────────────────────────────────────────────────────

.PHONY: run
run: ## [docker] Run full pipeline (all stages) for STATE/HOUR
	$(call check_data_dir)
	@mkdir -p $(OUTPUT_DIR)
	$(DOCKER_RUN) -c "bash /app/docker/run_pipeline.sh $(STATE) $(HOUR) $(N_PER_MODE)"

.PHONY: run-alabama
run-alabama: ## [docker] Run full pipeline for Alabama (16h)
	@$(MAKE) --no-print-directory run STATE=alabama HOUR=16h

.PHONY: run-montana
run-montana: ## [docker] Run full pipeline for Montana (16h)
	@$(MAKE) --no-print-directory run STATE=montana HOUR=16h

.PHONY: run-both
run-both: run-alabama run-montana ## [docker] Run full pipeline for Alabama and Montana

# ── Individual Stages ────────────────────────────────────────────────────────

.PHONY: patch
patch: ## [docker] Patch model JSON for PowerModels.jl compatibility
	$(call check_data_dir)
	@mkdir -p $(OUTPUT_DIR)/$(STATE)_$(HOUR)
	$(DOCKER_RUN) -c "\
	  julia --project=/app /app/docker/patch_model.jl \
	    /data/$(HOUR)/$(STATE)_model.json \
	    /output/$(STATE)_$(HOUR)/$(STATE)_model_patched.json"

.PHONY: solve
solve: ## [docker] Stage 2: Solve raw topology → cold-strict solvable JSON
	$(call check_data_dir)
	@mkdir -p $(OUTPUT_DIR)/$(STATE)_$(HOUR)
	$(DOCKER_RUN) -c "\
	  julia --project=/app /app/docker/patch_model.jl \
	    /data/$(HOUR)/$(STATE)_model.json \
	    /output/$(STATE)_$(HOUR)/$(STATE)_model_patched.json && \
	  julia --project=/app /app/solve_topo_json.jl \
	    /output/$(STATE)_$(HOUR)/$(STATE)_model_patched.json \
	    /output/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json"

.PHONY: export
export: ## [docker] Stage 3a: Export solvable JSON → single gridSFM .pyg.json
	$(DOCKER_RUN_LOCAL) -c "\
	  julia --project=/app /app/export_gridsfm_data.jl \
	    /output/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json \
	    /output/$(STATE)_$(HOUR)/$(STATE)_model.pyg.json"

.PHONY: perturb
perturb: ## [docker] Stage 3b: Generate perturbed scenarios (single state)
	@mkdir -p $(OUTPUT_DIR)/$(STATE)_$(HOUR)/scenarios
	$(DOCKER_RUN_LOCAL) -c "\
	  echo '/output/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json $(N_PER_MODE)' \
	    > /output/$(STATE)_$(HOUR)/grids_solvable.txt && \
	  julia --project=/app /app/gen_perturbed_data.jl \
	    /output/$(STATE)_$(HOUR)/grids_solvable.txt $(N_WORKERS) \
	    /output/$(STATE)_$(HOUR)/scenarios"

.PHONY: gen-bulk
gen-bulk: ## [docker] Stage 3b: Bulk scenario generation from GRID_LIST (multi-grid)
	$(if $(GRID_LIST),,$(error GRID_LIST is not set. Create a grid list file and pass GRID_LIST=/path/to/grids.txt))
	@mkdir -p $(OUTPUT_DIR)/scenarios
	docker run --rm \
	  -v $(OUTPUT_DIR):/output \
	  -v $(abspath $(GRID_LIST)):/tmp/grids.txt:ro \
	  $(IMAGE) -c "\
	  CPU_RANGE=0-$$(( $$(nproc) - 1 )) && \
	  bash /app/run_gen_gridsfm_data.sh $(N_WORKERS) /output/scenarios /tmp/grids.txt \
	    $$CPU_RANGE"

.PHONY: verify
verify: ## [docker] Verify: Round-trip check all scenarios for STATE/HOUR
	$(DOCKER_RUN_LOCAL) -c "\
	  SOLVABLE=/output/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json; \
	  CASE_DIR=/output/$(STATE)_$(HOUR)/scenarios/$(STATE)_model; \
	  found=0; \
	  for PYG in \$$CASE_DIR/*.pyg.json; do \
	    if [ ! -f \$$PYG ]; then \
	      continue; \
	    fi; \
	    found=1; \
	    julia --project=/app /app/solve_pyg_json.jl \$$SOLVABLE \$$PYG 2>&1 | grep RESOLVE; \
	  done; \
	  if [ \$$found -eq 0 ]; then \
	    echo \"(skip verify — no scenario .pyg.json files found in \$$CASE_DIR)\"; \
	  fi"

.PHONY: verify-one
verify-one: ## [docker] Verify a single scenario: PYG_FILE=<path relative to OUTPUT_DIR>
	$(if $(PYG_FILE),,$(error PYG_FILE is not set. Example: make verify-one PYG_FILE=alabama_16h/scenarios/alabama_model/loads_0001.pyg.json))
	$(DOCKER_RUN_LOCAL) -c "\
	  julia --project=/app /app/solve_pyg_json.jl \
	    /output/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json \
	    /output/$(PYG_FILE)"

# ── Integration Test ─────────────────────────────────────────────────────────

.PHONY: integration-test
integration-test: ## [docker] Run end-to-end integration test for STATE/HOUR
	$(call check_data_dir)
	@mkdir -p $(OUTPUT_DIR)/integration_test
	$(DOCKER_RUN) -c "\
	  julia --project=/app /app/docker/patch_model.jl \
	    /data/$(HOUR)/$(STATE)_model.json \
	    /output/integration_test/$(STATE)_model.json && \
	  bash /app/integration_test_all_components.sh \
	    /output/integration_test/$(STATE)_model.json \
	    /output/integration_test/$(STATE)"

# ── Interactive / Debug ──────────────────────────────────────────────────────

.PHONY: shell
shell: ## [docker] Open an interactive bash shell inside the container
	$(call check_data_dir)
	@mkdir -p $(OUTPUT_DIR)
	docker run --rm -it \
	  -v $(DATA_DIR):/data:ro \
	  -v $(OUTPUT_DIR):/output \
	  $(IMAGE)

.PHONY: julia-repl
julia-repl: ## [docker] Open a Julia REPL inside the container
	$(call check_data_dir)
	@mkdir -p $(OUTPUT_DIR)
	docker run --rm -it \
	  -v $(DATA_DIR):/data:ro \
	  -v $(OUTPUT_DIR):/output \
	  --entrypoint julia \
	  $(IMAGE) --project=/app

# =============================================================================
#  LOCAL TARGETS (no Docker — requires Julia installed on host)
# =============================================================================

# ── Install ──────────────────────────────────────────────────────────────────

.PHONY: local-install
local-install: ## [local] Install Julia packages (one-time setup)
	julia $(JULIA_PROJECT) -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
	@julia $(JULIA_PROJECT) -e 'using PowerModels, Ipopt, JuMP, JSON; println("✓ All packages loaded")'

.PHONY: local-check
local-check: ## [local] Verify Julia and packages are working
	@julia --version
	@julia $(JULIA_PROJECT) -e 'using PowerModels, Ipopt, JuMP, JSON; println("✓ All packages loaded")'

# ── Full Pipeline ────────────────────────────────────────────────────────────

.PHONY: local-run
local-run: ## [local] Run full pipeline (all stages) for STATE/HOUR
	$(call check_data_dir)
	@mkdir -p $(OUTPUT_DIR)/$(STATE)_$(HOUR)
	@# Patch
	julia $(JULIA_PROJECT) $(CURDIR)/docker/patch_model.jl \
	  $(DATA_DIR)/$(HOUR)/$(STATE)_model.json \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model_patched.json
	@# Solve
	julia $(JULIA_PROJECT) $(PIPE_DIR)/solve_topo_json.jl \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model_patched.json \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json
	@# Export
	julia $(JULIA_PROJECT) $(PIPE_DIR)/export_gridsfm_data.jl \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model.pyg.json
	@# Perturb
	@mkdir -p $(OUTPUT_DIR)/$(STATE)_$(HOUR)/scenarios
	@echo '$(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json $(N_PER_MODE)' \
	  > $(OUTPUT_DIR)/$(STATE)_$(HOUR)/grids_solvable.txt
	julia $(JULIA_PROJECT) $(PIPE_DIR)/gen_perturbed_data.jl \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/grids_solvable.txt $(N_WORKERS) \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/scenarios
	@# Verify
	@SOLVABLE=$(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json; \
	CASE_DIR=$(OUTPUT_DIR)/$(STATE)_$(HOUR)/scenarios/$(STATE)_model; \
	found=0; \
	for PYG in $$CASE_DIR/*.pyg.json; do \
	  if [ ! -f "$$PYG" ]; then \
	    continue; \
	  fi; \
	  found=1; \
	  julia $(JULIA_PROJECT) $(PIPE_DIR)/solve_pyg_json.jl "$$SOLVABLE" "$$PYG" 2>&1 | grep RESOLVE; \
	done; \
	if [ $$found -eq 0 ]; then \
	  echo "(skip verify — no scenario .pyg.json files found in $$CASE_DIR)"; \
	fi

.PHONY: local-run-alabama
local-run-alabama: ## [local] Run full pipeline for Alabama (16h)
	@$(MAKE) --no-print-directory local-run STATE=alabama HOUR=16h

.PHONY: local-run-montana
local-run-montana: ## [local] Run full pipeline for Montana (16h)
	@$(MAKE) --no-print-directory local-run STATE=montana HOUR=16h

.PHONY: local-run-both
local-run-both: local-run-alabama local-run-montana ## [local] Run full pipeline for Alabama and Montana

# ── Individual Stages ────────────────────────────────────────────────────────

.PHONY: local-patch
local-patch: ## [local] Patch model JSON for PowerModels.jl compatibility
	$(call check_data_dir)
	@mkdir -p $(OUTPUT_DIR)/$(STATE)_$(HOUR)
	julia $(JULIA_PROJECT) $(CURDIR)/docker/patch_model.jl \
	  $(DATA_DIR)/$(HOUR)/$(STATE)_model.json \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model_patched.json

.PHONY: local-solve
local-solve: ## [local] Stage 2: Solve raw topology → cold-strict solvable JSON
	$(call check_data_dir)
	@mkdir -p $(OUTPUT_DIR)/$(STATE)_$(HOUR)
	julia $(JULIA_PROJECT) $(CURDIR)/docker/patch_model.jl \
	  $(DATA_DIR)/$(HOUR)/$(STATE)_model.json \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model_patched.json
	julia $(JULIA_PROJECT) $(PIPE_DIR)/solve_topo_json.jl \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model_patched.json \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json

.PHONY: local-export
local-export: ## [local] Stage 3a: Export solvable JSON → single gridSFM .pyg.json
	julia $(JULIA_PROJECT) $(PIPE_DIR)/export_gridsfm_data.jl \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model.pyg.json

.PHONY: local-perturb
local-perturb: ## [local] Stage 3b: Generate perturbed scenarios (single state)
	@mkdir -p $(OUTPUT_DIR)/$(STATE)_$(HOUR)/scenarios
	@echo '$(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json $(N_PER_MODE)' \
	  > $(OUTPUT_DIR)/$(STATE)_$(HOUR)/grids_solvable.txt
	julia $(JULIA_PROJECT) $(PIPE_DIR)/gen_perturbed_data.jl \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/grids_solvable.txt $(N_WORKERS) \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/scenarios

.PHONY: local-gen-bulk
local-gen-bulk: ## [local] Stage 3b: Bulk scenario generation from GRID_LIST (multi-grid)
	$(if $(GRID_LIST),,$(error GRID_LIST is not set. Create a grid list file and pass GRID_LIST=/path/to/grids.txt))
	@mkdir -p $(OUTPUT_DIR)/scenarios
	CPU_RANGE="$(CPU_RANGE)"; \
	if [ -z "$$CPU_RANGE" ]; then \
	  if command -v nproc >/dev/null 2>&1; then \
	    ncpu=$$(nproc); \
	  elif command -v sysctl >/dev/null 2>&1; then \
	    ncpu=$$(sysctl -n hw.logicalcpu 2>/dev/null || true); \
	  else \
	    ncpu=$$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1); \
	  fi; \
	  case "$$ncpu" in ''|*[!0-9]*|0) ncpu=1 ;; esac; \
	  CPU_RANGE="0-$$((ncpu - 1))"; \
	fi; \
	bash $(PIPE_DIR)/run_gen_gridsfm_data.sh \
	  $(N_WORKERS) $(OUTPUT_DIR)/scenarios $(GRID_LIST) "$$CPU_RANGE"

.PHONY: local-verify
local-verify: ## [local] Verify: Round-trip check all scenarios for STATE/HOUR
	@SOLVABLE=$(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json; \
	CASE_DIR=$(OUTPUT_DIR)/$(STATE)_$(HOUR)/scenarios/$(STATE)_model; \
	found=0; \
	for PYG in $$CASE_DIR/*.pyg.json; do \
	  if [ ! -f "$$PYG" ]; then \
	    continue; \
	  fi; \
	  found=1; \
	  julia $(JULIA_PROJECT) $(PIPE_DIR)/solve_pyg_json.jl "$$SOLVABLE" "$$PYG" 2>&1 | grep RESOLVE; \
	done; \
	if [ $$found -eq 0 ]; then \
	  echo "(skip verify — no scenario .pyg.json files found in $$CASE_DIR)"; \
	fi

.PHONY: local-verify-one
local-verify-one: ## [local] Verify a single scenario: PYG_FILE=<path relative to OUTPUT_DIR>
	$(if $(PYG_FILE),,$(error PYG_FILE is not set. Example: make local-verify-one PYG_FILE=alabama_16h/scenarios/alabama_model/loads_0001.pyg.json))
	julia $(JULIA_PROJECT) $(PIPE_DIR)/solve_pyg_json.jl \
	  $(OUTPUT_DIR)/$(STATE)_$(HOUR)/$(STATE)_model.solvable.json \
	  $(OUTPUT_DIR)/$(PYG_FILE)

# ── Integration Test ─────────────────────────────────────────────────────────

.PHONY: local-integration-test
local-integration-test: ## [local] Run end-to-end integration test for STATE/HOUR
	$(call check_data_dir)
	@mkdir -p $(OUTPUT_DIR)/integration_test
	julia $(JULIA_PROJECT) $(CURDIR)/docker/patch_model.jl \
	  $(DATA_DIR)/$(HOUR)/$(STATE)_model.json \
	  $(OUTPUT_DIR)/integration_test/$(STATE)_model.json
	bash $(PIPE_DIR)/integration_test_all_components.sh \
	  $(OUTPUT_DIR)/integration_test/$(STATE)_model.json \
	  $(OUTPUT_DIR)/integration_test/$(STATE)

# ── Interactive ──────────────────────────────────────────────────────────────

.PHONY: local-julia-repl
local-julia-repl: ## [local] Open a Julia REPL with the project environment
	julia $(JULIA_PROJECT)

# =============================================================================
#  SHARED TARGETS (work for both Docker and local)
# =============================================================================

# ── Inspection ───────────────────────────────────────────────────────────────

.PHONY: list-states
list-states: ## List available states in the data directory
	$(call check_data_dir)
	@ls $(DATA_DIR)/$(HOUR)/*_model.json 2>/dev/null | \
	  sed 's|.*/||; s|_model\.json||' | sort

.PHONY: list-output
list-output: ## List output artifacts
	@find $(OUTPUT_DIR) -type f 2>/dev/null | sort

.PHONY: output-size
output-size: ## Show output directory sizes
	@du -sh $(OUTPUT_DIR)/*/ 2>/dev/null || echo "(no output yet)"

# ── Cleanup ──────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove all output artifacts
	docker run --rm -v $(OUTPUT_DIR):/output $(IMAGE) -c "rm -rf /output/*" 2>/dev/null; \
	rm -rf $(OUTPUT_DIR); true

.PHONY: clean-state
clean-state: ## Remove output for a specific STATE/HOUR
	docker run --rm -v $(OUTPUT_DIR):/output $(IMAGE) -c "rm -rf /output/$(STATE)_$(HOUR)" 2>/dev/null; \
	rm -rf $(OUTPUT_DIR)/$(STATE)_$(HOUR); true

.PHONY: clean-image
clean-image: ## Remove the Docker image
	docker rmi $(IMAGE) 2>/dev/null || true

.PHONY: clean-all
clean-all: clean clean-image ## Remove all output and the Docker image

# ── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@echo "topology_solver_pipeline (Docker + Local)"
	@echo ""
	@echo "Usage: make <target> [VAR=value ...]"
	@echo ""
	@echo "Variables (override with VAR=value):"
	@echo "  IMAGE       Docker image name          (default: topo_solver_pipe)"
	@echo "  DATA_DIR    Path to GridSFM_US_power_grid dir  (REQUIRED for solve/run)"
	@echo "  OUTPUT_DIR  Host output directory       (default: ./output)"
	@echo "  STATE       State name                  (default: alabama)"
	@echo "  HOUR        Operating hour (16h/04h)    (default: 16h)"
	@echo "  N_PER_MODE  Scenarios per perturb mode  (default: 1)"
	@echo "  N_WORKERS   Julia worker processes      (default: 2)"
	@echo "  GRID_LIST   Grid list file for gen-bulk (REQUIRED for gen-bulk)"
	@echo "  PYG_FILE    Scenario file for verify-one (relative to OUTPUT_DIR)"
	@echo ""
	@echo "Docker targets:"
	@grep -E '^[a-zA-Z_-]+:.*## \[docker\]' $(MAKEFILE_LIST) | \
	  awk -F ':.*## \\[docker\\] ' '{printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Local targets (no Docker):"
	@grep -E '^[a-zA-Z_-]+:.*## \[local\]' $(MAKEFILE_LIST) | \
	  awk -F ':.*## \\[local\\] ' '{printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Shared targets:"
	@grep -E '^[a-zA-Z_-]+:.*## [^[]' $(MAKEFILE_LIST) | \
	  grep -v '## \[' | \
	  awk -F ':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
