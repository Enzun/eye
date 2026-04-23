from PyQt5.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from .import_tab import ImportTab
from .inference_tab import InferenceTab
from .distribute_tab import DistributeTab
from .progress_tab import ProgressTab
from .export_tab import ExportTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("主PC管理ツール (Admin Manager)")
        self.resize(1200, 800)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout(self.central_widget)
        
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # 各タブの追加
        self.import_tab = ImportTab(self)
        self.inference_tab = InferenceTab(self)
        self.distribute_tab = DistributeTab(self)
        self.progress_tab = ProgressTab(self)
        self.export_tab = ExportTab(self)
        
        self.tabs.addTab(self.import_tab, "インポート")
        self.tabs.addTab(self.inference_tab, "AI予測")
        self.tabs.addTab(self.distribute_tab, "匿名化と配布")
        self.tabs.addTab(self.progress_tab, "進捗確認")
        self.tabs.addTab(self.export_tab, "学習用出力")
