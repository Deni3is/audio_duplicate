import sys
import os
import numpy as np
import librosa
import librosa.display
import datetime
from compare_two_audio_files import Model



from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout,
    QLabel, QPushButton, QWidget, QHBoxLayout, QComboBox,
    QListWidget, QMessageBox, QFrame
)
from PyQt6.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtGui import QColor


class WaveformCanvas(FigureCanvas):
    def __init__(self, title="Waveform", parent=None):
        fig = Figure(figsize=(5, 2), dpi=100)
        self.ax = fig.add_subplot(111)
        self.title = title
        super().__init__(fig)
        

    def plot_waveform(self, y, sr):
        self.ax.clear()
        self.ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='blue')
        self.ax.set_title(self.title)
        self.ax.set_xlabel("Time (s)")
        self.draw()


class SpectrogramCanvas(FigureCanvas):
    def __init__(self, title="Spectrogram", parent=None):
        fig = Figure(figsize=(5, 2), dpi=100)
        self.ax = fig.add_subplot(111)
        self.title = title
        super().__init__(fig)

    def plot_spectrogram(self, y, sr):
        self.ax.clear()
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, ax=self.ax, x_axis='time', y_axis='mel')
        self.ax.set_title(self.title)
        self.draw()


class AudioCompareApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Сравнение аудиозаписей")
        self.setGeometry(100, 100, 1200, 900)

        self.audio_files = []
        self.audio_dir = ""
        self.model:Model = Model()

        self.init_ui()
        self.update_clock()

    def init_ui(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.clock = QLabel()
        self.clock.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.clock)

        timer = QTimer(self)
        timer.timeout.connect(self.update_clock)
        timer.start(1000)

        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("Выберите папку с аудио:")
        self.select_folder_btn = QPushButton("Выбрать папку")
        self.select_folder_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.select_folder_btn)
        layout.addLayout(folder_layout)

        file_select_layout = QHBoxLayout()
        self.combo1 = QComboBox()
        self.combo2 = QComboBox()
        file_select_layout.addWidget(QLabel("Аудио 1:"))
        file_select_layout.addWidget(self.combo1)
        file_select_layout.addWidget(QLabel("Аудио 2:"))
        file_select_layout.addWidget(self.combo2)

        self.compare_btn = QPushButton("Сравнить")
        self.compare_btn.clicked.connect(self.compare_audio)
        file_select_layout.addWidget(self.compare_btn)
        layout.addLayout(file_select_layout)

        wave_1 = QHBoxLayout()
        wave_2 = QHBoxLayout()

        self.waveform1 = WaveformCanvas("Waveform Audio 1")
        self.waveform2 = WaveformCanvas("Waveform Audio 2")
        self.spectrogram1 = SpectrogramCanvas("Spectrogram Audio 1")
        self.spectrogram2 = SpectrogramCanvas("Spectrogram Audio 2")
        wave_1.addWidget(self.waveform1)
        wave_1.addWidget(self.spectrogram1)
        wave_2.addWidget(self.waveform2)
        wave_2.addWidget(self.spectrogram2)



        layout.addLayout(wave_1)
        layout.addLayout(wave_2)

        self.result_frame = QFrame()
        self.result_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.result_layout = QVBoxLayout()
        self.result_label = QLabel("Результат: —")
        self.result_label.setStyleSheet("font-size: 18px; padding: 10px;")
        self.result_layout.addWidget(self.result_label)
        self.save_report_btn = QPushButton("Сохранить отчет")
        self.save_report_btn.clicked.connect(self.save_report)
        
        self.result_frame.setLayout(self.result_layout)
        layout.addWidget(self.result_frame)

        mass_layout = QHBoxLayout()
        mass1_layout = QHBoxLayout()
        mass2_layout = QHBoxLayout()
        
        self.mass_compare_btn = QPushButton("Поиск дубликатов в папке")
        self.mass_compare_btn.clicked.connect(self.mass_compare)
        self.result_list = QListWidget()

        mass1_layout.addWidget(self.mass_compare_btn)
        mass2_layout.addWidget(self.save_report_btn)
        
        mass_layout.addWidget(self.result_list)
        
        layout.addLayout(mass1_layout)        
        layout.addLayout(mass_layout)
        layout.addLayout(mass2_layout)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def update_clock(self):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.clock.setText(f"Текущее время: {now}")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбрать папку с аудио")
        if folder:
            self.audio_dir = folder
            self.audio_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
            self.combo1.clear()
            self.combo2.clear()
            self.combo1.addItems(self.audio_files)
            self.combo2.addItems(self.audio_files)

    def load_audio(self, filename):
        path = os.path.join(self.audio_dir, filename)
        y, sr = librosa.load(path, sr=None)
        return y, sr

    def compare_audio(self):
        file1 = self.combo1.currentText()
        file2 = self.combo2.currentText()
        if not file1 or not file2:
            return
        try:   
            y1, sr1 = self.load_audio(file1)
        except Exception as ex:
            QMessageBox.warning(self, "Внимание", "Файл 1 имеет неверный формат или отсутствует.")
            return
        
        try:
            y2, sr2 = self.load_audio(file2)
        except Exception as ex:
            QMessageBox.warning(self, "Внимание", "Файл 2 имеет неверный формат или отсутствует.")
            return
         
        self.waveform1.plot_waveform(y1, sr1)
        self.waveform2.plot_waveform(y2, sr2)
        self.spectrogram1.plot_spectrogram(y1, sr1)
        self.spectrogram2.plot_spectrogram(y2, sr2)

        similarity = self.model.inference(os.path.join(self.audio_dir, file1),
                                          os.path.join(self.audio_dir, file2))
        if similarity > 0.75:
            quality = "Высокая вероятность нечеткого дубликата или дубликата"
            color = "#05bf11"
        else:
            quality = "Не дубликат"
            color = "#840707"

        self.result_frame.setStyleSheet(f"background-color: {color}; border: 1px solid #ccc;")
        self.result_label.setText(
            f"Сходство между \"{file1}\" и \"{file2}\": {similarity * 100:.1f}% — {quality}"
        )

    def save_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчет", "отчет.txt")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                if self.result_label.text() != "":
                    f.write(self.result_label.text() + '\n')
                if self.result_list.count() > 0:
                    for i in range(self.result_list.count()):
                        item = self.result_list.item(i)
                        f.write(item.text() + '\n')
            QMessageBox.information(self, "Отчет сохранен", f"Отчет сохранен в: {path}")


    def mass_compare(self):
        if not self.audio_files:
            QMessageBox.warning(self, "Внимание", "Сначала выберите папку с аудиофайлами.")
            return

        self.result_list.clear()
        pairs_checked = set()
        for i in range(len(self.audio_files)):
            for j in range(i + 1, len(self.audio_files)):
                f1 = self.audio_files[i]
                f2 = self.audio_files[j]
                score = self.model.inference(self.audio_dir +'/'+ f1,self.audio_dir +'/'+ f2)
                if score is None:
                    continue
                if score > 0.7:
                    self.result_list.addItem(f"{f1} <=> {f2} — {score * 100:.1f}%")
                pairs_checked.add((f1, f2))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioCompareApp()
    window.show()
    sys.exit(app.exec())
