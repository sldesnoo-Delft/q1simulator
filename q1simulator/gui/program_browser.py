import logging
import os

from qtpy import QtCore, QtWidgets

from q1simulator.gui.program_details import ProgramDetailsWidget
from q1simulator.gui.qt_utils import (
    qt_log_exception,
    qt_init,
    qt_create_app,
    qt_run_app,
    qt_set_darkstyle,
)


logger = logging.getLogger(__name__)


_app = None


class Q1ProgramBrowser(QtWidgets.QMainWindow):
    """
    Q1Program browser.
    """

    _WINDOW_TITLE: str = "Q1Program Browser"

    def __init__(
        self,
        path: str | None = None,
        gui_style: str | None = None
    ):
        """Creates program browser

        Args:
            path: base directory. If None uses current working directory.
            gui_style: if "dark" uses dark style, otherwise normal style.
        """
        global _app
        logger.debug("Init program browser")

        qt_app_runing = qt_init()
        if not qt_app_runing:
            # note: store reference to avoid garbage collection.
            # reference is also used to restart browser 2nd time in Python console.
            _app = qt_create_app()

        if gui_style == "dark":
            qt_set_darkstyle()

        super().__init__()

        if path is None:
            path = os.getcwd()

        self._path = path

        self.setWindowTitle(self._WINDOW_TITLE)
        self.resize(860, 600)
        self.create_ui()
        self.fill_program_list()
        self.show()

        if _app is not None:
            qt_run_app(_app)

    def create_ui(self):
        content = QtWidgets.QWidget()
        self._list_widget = QtWidgets.QListWidget()
        self._list_widget.setMinimumWidth(300)
        self._list_widget.itemSelectionChanged.connect(self._show_program_info)

        self._program_details = ProgramDetailsWidget(self)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._list_widget)
        layout.addWidget(self._program_details, 1)

        content.setLayout(layout)
        self.setCentralWidget(content)

    @QtCore.Slot()
    def _show_program_info(self):
        selected_items = self._list_widget.selectedItems()
        if not selected_items:
            return
        item = selected_items[0]
        path = item.data(QtCore.Qt.UserRole)
        self._program_details.show_details(path)

    def fill_program_list(self):
        w = self._list_widget
        w.clear()
        for entry in os.scandir(self._path):
            if not entry.is_dir():
                continue
            i = w.count()
            w.addItem(entry.name)
            item = w.item(i)
            item.setData(QtCore.Qt.UserRole, entry.path)

    def _show_error_message(self, title, message):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle("Q1ProgramBrowser: " + title)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()


if __name__ == "__main__":
    ui = Q1ProgramBrowser(r"C:\measurements\qblox_programs", gui_style="dark")
