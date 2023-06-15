import logging
import pytest
from utils import container_run

@pytest.mark.build
@pytest.mark.mlir
def test_mlir_efficiency(target, mlir_env, get_dataset):  # del get_imagenet_val, get_cifar100, get_coco2017_val
    if not mlir_env['case_list']:
        logging.info(f'Skip efficiency test')
        return

    container_run(mlir_env, f'cd /workspace/model-zoo && python3 -m tpu_perf.build {mlir_env["case_list"]} --outdir mlir_out_{target} --target {target} --mlir')
