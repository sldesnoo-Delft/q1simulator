import logging
import os
import threading
import traceback

import pyqtgraph as pg
from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QMessageBox

try:
    import IPython.lib.guisupport as gs
    from IPython import get_ipython
except Exception:
    def get_ipython():
        pass


logger = logging.getLogger(__name__)

is_wrapped = threading.local()
is_wrapped.val = False


def qt_log_exception(func):
    ''' Decorator to log exceptions.
    Exceptions are logged and raised again.
    Decorator is designed to be used around functions being called as
    QT event handlers, because QT doesn't report the exceptions.
    Note:
        The decorated method/function cannot be used with
        functools.partial.
    '''

    def wrapped(*args, **kwargs):
        if is_wrapped.val:
            return func(*args, **kwargs)
        else:
            is_wrapped.val = True
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.error('Exception in GUI', exc_info=True)
                raise
            finally:
                is_wrapped.val = False

    return wrapped


def qt_show_exception(message, ex, extra_line: str = None):
    logger.error(message, exc_info=ex)
    text = message
    if extra_line:
        text += "\n" + extra_line
    text += f"\n{type(ex).__name__}: {ex}"
    msg = QMessageBox(
        QMessageBox.Critical,
        "QT-DataViewer: " + message,
        text,
        QMessageBox.Ok,
        )
    msg.setDetailedText("\n".join(traceback.format_exception(ex)))
    msg.setStyleSheet("QTextEdit{min-width:600px}")
    msg.exec_()


def qt_show_error(title, message):
    logger.error(message)
    msg = QMessageBox(
        QMessageBox.Critical,
        "QT-DataViewer: " + title,
        message,
        QMessageBox.Ok,
        )
    msg.exec_()


_qt_app = None


def qt_init(style: str | None = None) -> bool:
    '''Tries to start the QT application if not yet started.
    Most of the cases the QT backend is already started
    by IPython, but sometimes it is not.

    Returns:
        False if QT backend is not running.
    '''
    # application reference must be held in global scope
    global _qt_app

    if _qt_app is None:

        # Set attributes for proper scaling when display scaling is not equal to 100%
        # This should be done before QApplication is started.
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

        ipython = get_ipython()

        if ipython:
            if not gs.is_event_loop_running_qt4():
                if any('SPYDER' in name for name in os.environ):
                    raise Exception('Configure QT5 in Spyder -> Preferences -> IPython Console -> Graphics -> Backend')
                else:
                    print('Warning Qt5 not configured for IPython console. Activating it now.')
                    ipython.run_line_magic('gui', 'qt5')

            _qt_app = QtCore.QCoreApplication.instance()
            if _qt_app is None:
                logger.debug('Create Qt application')
                _qt_app = QtWidgets.QApplication([])
            else:
                logger.debug('Qt application already created')
        else:
            _qt_app = QtCore.QCoreApplication.instance()

    if style == "dark":
        qt_set_darkstyle()

    # Only change if still default settings. Do not change is already changed
    if pg.getConfigOption('foreground') == 'd' and pg.getConfigOption('background') == 'k':
        pg.setConfigOption('background', None)
        pg.setConfigOption('foreground', 'k')

    return _qt_app is not None


def qt_set_darkstyle():
    import qdarkstyle
    import pyqtgraph as pg

    qt_app = QtCore.QCoreApplication.instance()
    if qt_app is None:
        return
    dark_stylesheet = qdarkstyle.load_stylesheet()
    # patch qdarkstyle for cropped x-label on 2D graphics.
    dark_stylesheet += r'''
QGraphicsView {
    padding: 0px;
}
'''
    qt_app.setStyleSheet(dark_stylesheet)
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'gray')


def qt_create_app() -> QtCore.QCoreApplication:
    logger.info("Create Qt application")
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication([])
    return app


def qt_run_app(app):
    logger.info("Run Qt Application")
    app.exec()
    logger.info("Qt Application event loop exited")
