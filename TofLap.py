import sys
import time
import re
import serial
import cv2
import threading

from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


# ToF SENSOR THREAD (Z HEIGHT)

class ToFWorker(QThread):
    z_updated = pyqtSignal(float, int)

    def __init__(self, port='/dev/ttyACM1', baud=115200):
        super().__init__()
        self.port = port
        self.baud = baud
        self.running = True

    def run(self):
        ser = serial.Serial(self.port, self.baud, timeout=1)
        time.sleep(2)

        while self.running:
            raw_bytes = ser.readline()
            if not raw_bytes:
                continue

            line = raw_bytes.decode("utf-8", errors="ignore").strip()

            match = re.search(r"Z Height:\s*([0-9.]+)", line)
            if match:
                z = float(match.group(1))
                timestamp = int(time.time() * 1000)
                self.z_updated.emit(z, timestamp)

        ser.close()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


# CAMERA THREAD

class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage, float, float)  # image, focus, z

    def __init__(self):
        super().__init__()
        self.running = True
        self._z_lock = threading.Lock()
        self.current_z = None
        self._init_camera()

    def _init_camera(self):
        from picamera2 import Picamera2

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "XBGR8888", "size": (640, 480)},
            buffer_count=4,
        )
        self.picam2.configure(config)
        self.picam2.start()

    def set_z(self, z):
        with self._z_lock:
            self.current_z = z

    def run(self):
        while self.running:
            frame = self.picam2.capture_array()

            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Focus metric (Laplacian variance)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            focus = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Convert to QImage
            h, w, ch = frame.shape
            qt_image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            qt_image = qt_image.copy()

            with self._z_lock:
                z = self.current_z if self.current_z is not None else -1

            self.frame_ready.emit(qt_image, focus, z)

            self.msleep(10)

    def stop(self):
        self.running = False
        try:
            self.picam2.stop()
        except Exception:
            pass
        self.quit()
        self.wait()


# MAIN GUI APPLICATION

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Viewer")

        self.label = QLabel("Starting...")
        self.label.setFixedSize(640, 480)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.current_z = None

        # Threads
        self.camera = CameraWorker()
        self.tof = ToFWorker()

        # Connections
        self.tof.z_updated.connect(self.update_z)
        self.camera.frame_ready.connect(self.update_frame)

        self.camera.start()
        self.tof.start()

    def update_z(self, z, timestamp):
        self.current_z = z
        self.camera.set_z(z)

    def update_frame(self, qimage, focus, z):
        self.label.setPixmap(QPixmap.fromImage(qimage))
        print(f"\rZ: {z:.2f} mm | Focus: {focus:.2f}    ", end="", flush=True)

    def closeEvent(self, event):
        self.camera.stop()
        self.tof.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())