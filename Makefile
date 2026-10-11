.PHONY: test test-root-io test-shell test-cpp test-bindings test-all ci

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
	bash tests/test_run_ml_iterative.sh
	bash tests/test_run_surface_helix_experiment.sh
	bash tests/test_plot_residuals_static.sh

test-cpp:
	cmake -S AlignDerivLib -B build/align-deriv-cpp -DALIGNDERIV_BUILD_PYTHON_PYBIND11=OFF -DALIGNDERIV_BUILD_PYTHON=OFF -DALIGNDERIV_BUILD_TESTS=ON
	cmake --build build/align-deriv-cpp -j2
	ctest --test-dir build/align-deriv-cpp --output-on-failure

test-bindings:
	cmake -S AlignDerivLib -B build/align-deriv-bindings -DALIGNDERIV_BUILD_PYTHON_PYBIND11=ON -DALIGNDERIV_BUILD_PYTHON=OFF -DALIGNDERIV_BUILD_TESTS=OFF
	cmake --build build/align-deriv-bindings -j2
	PYTHONPATH=build/align-deriv-bindings python -m pytest AlignDerivLib/tests/test_bindings.py

test-all: test test-shell

ci: test-all
