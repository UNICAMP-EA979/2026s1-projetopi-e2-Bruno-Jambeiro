import sys
import numpy as np
from PIL import Image

from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pywt


class Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FFT / DWT Viewer")
        self.resize(1300, 900)

        self.image = None
        self.mode = "FFT"  # or "DWT"

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QHBoxLayout()

        btn_open = QPushButton("Abrir imagem")
        btn_open.clicked.connect(self.load_image)
        top.addWidget(btn_open)

        self.mode_btn = QPushButton("Modo: FFT")
        self.mode_btn.clicked.connect(self.toggle_mode)
        top.addWidget(self.mode_btn)

        layout.addLayout(top)

        self.fig = Figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self._clear()

    def toggle_mode(self):
        self.mode = "DWT" if self.mode == "FFT" else "FFT"
        self.mode_btn.setText(f"Modo: {self.mode}")

        if self.image is not None:
            self.process()
            self.update_plot()

    def _clear(self):
        self.fig.clear()
        self.canvas.draw()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagem", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )

        if not path:
            return

        try:
            img = Image.open(path).convert("L")
            self.image = np.array(img, dtype=float)

            self.process()
            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "Erro", str(e))

    def process(self):
        # FFT
        fft = np.fft.fft2(self.image)
        fft_shift = np.fft.fftshift(fft)

        self.mag = np.log1p(np.abs(fft_shift))
        self.phase = np.angle(fft_shift)

        recon = np.fft.ifft2(np.fft.ifftshift(fft_shift))
        self.recon = np.real(recon)

        # DWT
        # retorna (cA, (cH, cV, cD)) :contentReference[oaicite:0]{index=0}
        coeffs = pywt.dwt2(self.image, 'haar')
        self.LL, (self.LH, self.HL, self.HH) = coeffs

    def update_plot(self):
        self.fig.clear()

        if self.mode == "FFT":
            axes = [self.fig.add_subplot(2, 2, i+1) for i in range(4)]

            axes[0].imshow(self.image, cmap='gray')
            axes[0].set_title("Imagem")

            axes[1].imshow(self.mag, cmap='gray')
            axes[1].set_title("FFT Magnitude")

            axes[2].imshow(self.phase, cmap='gray')
            axes[2].set_title("FFT Fase")

            axes[3].imshow(self.recon, cmap='gray')
            axes[3].set_title("Reconstrução")

        else:
            axes = [self.fig.add_subplot(2, 2, i+1) for i in range(4)]

            axes[0].imshow(self.image, cmap='gray')
            axes[0].set_title("Imagem")

            axes[1].imshow(self.LL, cmap='gray')
            axes[1].set_title("LL (low freq)")

            axes[2].imshow(self.LH, cmap='gray')
            axes[2].set_title("LH (horizontal)")

            # combinar HL + HH pra não desperdiçar espaço
            combined = np.hstack([self.HL, self.HH])
            axes[3].imshow(np.log1p(np.abs(combined)), cmap='gray')
            axes[3].set_title("HL | HH")

        for ax in axes:
            ax.axis("off")

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Viewer()
    w.show()
    sys.exit(app.exec())