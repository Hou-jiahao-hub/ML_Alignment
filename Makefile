.PHONY: test test-root-io test-shell test-all ci

# CLI subprocesses that import torch first otherwise load the system libstdc++,
# which lacks the CXXABI awkward_cpp needs.
ifdef CONDA_PREFIX
test test-root-io test-shell: export LD_LIBRARY_PATH := $(CONDA_PREFIX)/lib$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH))
endif

test:
	python -m pytest tests

test-root-io:
	timeout 120s python -m pytest tests -m root_io

test-shell:
	bash tests/test_run_surface_helix_experiment.sh
	bash tests/test_plot_residuals_static.sh

test-all: test test-shell

ci: test-all
