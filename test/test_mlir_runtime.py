import logging
import utils
import pytest
import subprocess
import platform

@pytest.mark.runtime
@pytest.mark.mlir
def test_mlir_runtime(target, runtime_dependencies, mlir_runtime):
    if not mlir_runtime['case_list']:
        logging.info(f'Skip efficiency test')
        return

    cmd = f'python3 -m tpu_perf.run {mlir_runtime["case_list"]} --outdir mlir_out_{target} --target {target} --mlir'
    cmd += utils.get_devices_opt()
    logging.info(cmd)
    subprocess.run(cmd, shell=True, check=True)

@pytest.mark.skipif(platform.machine() == 'aarch64', reason='Aarch64 machines do not test model precision!')
@pytest.mark.runtime
@pytest.mark.mlir
def test_mlir_precision(target, precision_dependencies, mlir_runtime, get_imagenet_val, get_cifar100, get_coco2017_val, get_dataset):
    if not mlir_runtime['case_list']:
        logging.info(f'Skip precision test')
        return

    cmd = f'python3 -m tpu_perf.precision_benchmark {mlir_runtime["case_list"]} --outdir mlir_out_{target} --target {target} --mlir'
    cmd += utils.get_devices_opt()
    logging.info(cmd)
    subprocess.run(cmd, shell=True, check=True)