from qblox_instruments import __version__ as qblox_version_str
from packaging.version import Version
from q1simulator import __version__ as q1simulator_version


def check_qblox_instrument_version():
    if qblox_version > Version('0.11.1'):
        print(f'WARNING Q1Simulator {q1simulator_version} simulates up to qblox_instruments v0.11.1, '
              f'but qblox_instruments version {qblox_version} is installed')
    elif qblox_version < Version('0.9'):
        raise Exception('Q1Simulator {q1simulator_version} expects qblox_instruments version v0.9+')


qblox_version = Version(qblox_version_str)
