import logging
import os
import re
import subprocess

def container_run(nntc_env, cmd):
    container = nntc_env['container']
    logging.info(cmd)
    ret, output = container.exec_run(
        f'bash -c "{cmd}"',
        tty=True)
    output = output.decode()
    if output:
        logging.info(f'------>\n{output}')
    m = re.search('(?<=please check ).+\\.log', output)
    if m:
        log_fn = m.group(0).replace('/workspace', '..')
        cmd_fn = log_fn.replace('.log', '.cmd')
        with open(cmd_fn) as f:
            logging.info(f'cat {cmd_fn}\n{f.read()}')
        with open(log_fn) as f:
            logging.info(f'cat {log_fn}\n{f.read()}')

    assert ret == 0

def get_devices_opt():
    if 'DEVICES' in os.environ:
        return ' --devices ' + os.environ['DEVICES']
    else:
        return ''
