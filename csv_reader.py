import csv
import sys
from PyQt5.QtWidgets import QApplication, \
QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QGraphicsView, QGraphicsScene
from PyQt5.QtCore import Qt

class CSVData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.field_names = []

    def load_data(self):
        with open(self.file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i==0:
                    self.field_names = row
                else:
                    self.data.append(row)

    def get_field_names(self):
        return self.field_names

    def get_data(self):
        return self.data

class CSVTable(QGraphicsView):
    def __init__(self, csv_file):
        super().__init__()

        # 创建一个ExperimentCSV对象来加载CSV数据
        self.csv_data = CSVData(csv_file)
        self.csv_data.load_data()
        
        # 创建一个表格视图并设置列数和行数
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.csv_data.get_field_names()))
        self.table.setRowCount(len(self.csv_data.get_data()))

        # 设置表格的标题
        scale = QApplication.devicePixelRatio()
        self.table.setHorizontalHeaderLabels(self.csv_data.get_field_names())

        # 将CSV数据填充到表格中
        for i, row in enumerate(self.csv_data.get_data()):
            for j, col in enumerate(row):
                item = QTableWidgetItem(col)
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.table.setItem(i, j, item)
       
        #    with open(csv_file, 'r', encoding='utf-8') as file:
        #        reader = csv.reader(file)
        #        for row_num, row_data in enumerate(reader):
        #            self.table.insertRow(row_num)
        #            for col_num, col_data in enumerate(row_data):
        #                item = QTableWidgetItem(col_data)
        #                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        #                self.table.setItem(row_num, col_num, item)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    csv_file = 'log_vgg.csv'  # 替换为你的CSV文件路径
    widget = CSVTable(csv_file)
    widget.setWindowTitle('CSV阅读器')
    widget.show()

    sys.exit(app.exec_())
   
# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
# from CSVTable import CSVTable

# class MainWindow(QMainWindow):
#    def __init__(self):
#        super().__init__()

#        self.setWindowTitle('CSV阅读器')

#        csv_file = 'example.csv'  # 替换为你的CSV文件路径
#        self.csv_table = CSVTable(csv_file)

#        layout = QVBoxLayout()
#        layout.addWidget(self.csv_table)

#        container = QWidget()
#        container.setLayout(layout)
#        self.setCentralWidget(container)

# if __name__ == '__main__':
#    app = QApplication(sys.argv)

#    window = MainWindow()
#    window.show()

#    sys.exit(app.exec_())