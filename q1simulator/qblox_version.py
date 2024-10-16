from qblox_instruments import __version__ as qblox_version_str
from packaging.version import Version
from q1simulator import __version__ as q1simulator_version


def check_qblox_instrument_version():
    min_version = "0.11.0"
    max_version = "0.14.1"
    if qblox_version < Version(min_version):
        raise Exception(f'Q1Simulator {q1simulator_version} requires qblox_instruments v{min_version}+')
    if qblox_version > Version(max_version):
        print(f'WARNING Q1Simulator {q1simulator_version} simulates up to qblox_instruments v{max_version}, '
              f'but qblox_instruments version {qblox_version} is installed')


qblox_version = Version(qblox_version_str)
