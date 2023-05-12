# compilation configuration
SOURCE_DIR=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
BUILD_DIR=$(realpath $(SOURCE_DIR)/../release_build)
INSTALL_DIR=$(realpath $(SOURCE_DIR)/../release_install)
BUILD_TYPE=Release
PYTHON_EXECUTABLE=$(shell which python)
NUM_COMPILE_JOBS=$(shell nproc)

# ===================================
# Do not modify the following lines
# ===================================

ENTRY_POINT=$(INSTALL_DIR)/openparf.py

$(info )
$(info =============================)
$(info * Compilation Configuration *)
$(info =============================)
$(info SOURCE_DIR=$(SOURCE_DIR))
$(info BUILD_DIR=$(BUILD_DIR))
$(info INSTALL_DIR=$(INSTALL_DIR))
$(info BUILD_TYPE=$(BUILD_TYPE))
$(info PYTHON_EXECUTABLE=$(PYTHON_EXECUTABLE))
$(info NUM_COMPILE_JOBS=$(NUM_COMPILE_JOBS))
$(info )

.PHONY: all, example, test, regression

all:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake $(SOURCE_DIR) -DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DUSE_CCACHE=1 -DPYTHON_EXECUTABLE=$(PYTHON_EXECUTABLE)
	cd $(BUILD_DIR) && make -j$(NUM_COMPILE_JOBS)
	cd $(BUILD_DIR) && make install

test: all
	cd $(BUILD_DIR) && make test

doc: all

regression: all

run: all
	cd $(INSTALL_DIR) && $(PYTHON_EXECUTABLE) $(ENTRY_POINT) --config $(SOURCE_DIR)/unittest/regression/ispd2017_flexshelf/CLK-FPGA01_flexshelf.json

run/debug: all
	cd $(INSTALL_DIR) && $(PYTHON_EXECUTABLE) -m pdb $(ENTRY_POINT) --config $(SOURCE_DIR)/unittest/regression/ispd2017_flexshelf/CLK-FPGA01_flexshelf.json

run/regression: all
	cd $(INSTALL_DIR) && $(SOURCE_DIR)/scripts/run_ispd2017_benchmark.sh $(shell date +%Y%m%d_%H%M%S)
