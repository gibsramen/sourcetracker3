import logging
import warnings

import cmdstanpy


logger = logging.getLogger("sourcetracker3")
formatter = logging.Formatter(
    "[%(asctime)s - %(name)s - %(levelname)s] ::  %(message)s"
)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(sh)

warnings_logger = logging.getLogger("py.warnings")
warnings_logger.addHandler(sh)

cmdstanpy_logger = cmdstanpy.utils.get_logger()
cmdstanpy_logger.setLevel(logging.DEBUG)
for h in cmdstanpy_logger.handlers:
    h.setFormatter(formatter)
