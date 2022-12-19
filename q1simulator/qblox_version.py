
from qblox_instruments import __version__ as qblox_version_str
from packaging.version import Version
from q1simulator import __version__ as q1simulator_version

def check_qblox_instrument_version():
    if qblox_version >= Version('0.9'):
        print(f'WARNING Q1Simulator {q1simulator_version} simulates qblox_instruments v0.8.x, '
              f'but qblox_instruments version {qblox_version} is installed')
    elif qblox_version < Version('0.8'):
        raise Exception('Q1Simulator v0.8.x expects qblox_instruments version  v0.8.y')

qblox_version = Version(qblox_version_str)
