from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWizard, QWizardPage)

from prompt_master.core.paths import AppPaths
from prompt_master.inference.device_detection import (QUANTIZATIONS, detect_gpus,
    recommended_quantization, vram_shortfall_mb)
from prompt_master.provisioning import installer


class SetupWizard(QWizard):
    """Provision and validate a complete local runtime; no state is saved early.

    The three questions asked here — directory, GPU, quantization — are the same
    three the console installer asks, and both hand off to
    ``provisioning.installer`` for everything after the last answer, so the two
    front ends cannot provision differently.
    """

    def __init__(self, paths: AppPaths, parent=None):
        super().__init__(parent); self.paths=paths; self.gpus=[]; self.completed=False
        self.setWindowTitle("Models and Hardware Setup"); self.setOption(QWizard.NoBackButtonOnStartPage)
        self._location_page(); self._hardware_page(); self._model_page(); self._install_page()
        self.currentIdChanged.connect(self._page_changed)

    def _location_page(self):
        page=QWizardPage(); page.setTitle("Installation directory")
        layout=QVBoxLayout(page); layout.addWidget(QLabel("Runtime, model, projector, cache, logs, and setup state remain under this directory."))
        row=QHBoxLayout(); self.location=QLineEdit(str(self.paths.root)); choose=QPushButton("Browse…"); choose.clicked.connect(self._browse); row.addWidget(self.location); row.addWidget(choose); layout.addLayout(row); self.addPage(page)

    def _hardware_page(self):
        page=QWizardPage(); page.setTitle("GPU selection"); layout=QVBoxLayout(page)
        self.hardware_status=QLabel("Open this page to scan with nvidia-smi."); self.hardware_status.setWordWrap(True)
        self.gpu=QComboBox(); layout.addWidget(self.hardware_status); layout.addWidget(self.gpu); self.addPage(page)

    def _model_page(self):
        page=QWizardPage(); page.setTitle("Model quality"); form=QFormLayout(page)
        self.quant=QComboBox(); self.quant.addItems(list(QUANTIZATIONS)); self.recommendation=QLabel(); self.recommendation.setWordWrap(True)
        form.addRow("GGUF quantization",self.quant); form.addRow(self.recommendation)
        self.quant.currentTextChanged.connect(self._describe_quant); self.addPage(page)

    def _install_page(self):
        page=QWizardPage(); page.setTitle("Download, verify, and validate"); layout=QVBoxLayout(page)
        text=QLabel("Finish downloads the pinned llama.cpp runtime, model, and matching vision projector; verifies size and SHA-256; then validates one text and one image request."); text.setWordWrap(True); layout.addWidget(text)
        self.progress=QProgressBar(); self.install_status=QLabel("Ready"); self.install_status.setWordWrap(True); layout.addWidget(self.progress); layout.addWidget(self.install_status); self.addPage(page)

    def _browse(self):
        value=QFileDialog.getExistingDirectory(self,"Installation directory",self.location.text())
        if value: self.location.setText(value)

    def _page_changed(self,index):
        if index == 1:
            self.gpu.clear()
            try: self.gpus=detect_gpus()
            except Exception as exc: self.gpus=[]; self.hardware_status.setText(f"GPU scan failed: {exc}"); return
            for gpu in self.gpus: self.gpu.addItem(f"{gpu.name} — {gpu.memory_total_mb} MiB — {gpu.uuid}",gpu)
            self.hardware_status.setText(
                f"Found {len(self.gpus)} CUDA GPU(s)." if self.gpus
                else "No CUDA GPU found. This application requires an NVIDIA GPU.")
        elif index == 2 and self.gpu.currentData():
            self.quant.setCurrentText(recommended_quantization(self.gpu.currentData())); self._describe_quant()

    def _describe_quant(self,*_):
        """Recommendation, download size and the VRAM warning, together.

        The warning is shown rather than enforced: a card below the threshold
        still installs and runs, it just spills layers to system RAM, and that
        trade is the user's to make."""
        gpu=self.gpu.currentData()
        if gpu is None: return
        quant=self.quant.currentText()
        lines=[f"Recommended for {gpu.name} ({gpu.memory_total_mb} MiB): {recommended_quantization(gpu)}",
               f"Download: {installer.format_download_size(gpu,quant)}."]
        shortfall=vram_shortfall_mb(gpu,quant)
        if shortfall: lines.append(f"Warning: {quant} wants roughly {shortfall} MiB more VRAM than this card reports. It will still install, but llama.cpp will spill layers to system RAM and generation will be slow.")
        self.recommendation.setText("\n".join(lines))

    def validateCurrentPage(self):
        if self.currentId() == 0:
            root=Path(self.location.text()).expanduser().resolve()
            try: root.mkdir(parents=True,exist_ok=True); probe=root/".write-test"; probe.write_bytes(b""); probe.unlink()
            except OSError as exc: QMessageBox.critical(self,"Invalid directory",str(exc)); return False
            self.paths=AppPaths(root); self.paths.create_managed_dirs()
        elif self.currentId() == 1 and not self.gpu.currentData():
            QMessageBox.critical(self,"No GPU selected","Select an NVIDIA GPU. nvidia-smi reported none."); return False
        elif self.currentId() == 3 and not self.completed:
            try: self._provision()
            except Exception as exc: self.install_status.setText(f"Setup failed: {exc}"); QMessageBox.critical(self,"Setup failed",str(exc)); return False
        return super().validateCurrentPage()

    def _provision(self):
        installer.provision(self.paths,self.gpu.currentData(),self.quant.currentText(),
                            on_status=self._status,on_progress=self._progress)
        self.progress.setValue(100); self.completed=True

    def _status(self,message):
        self.install_status.setText(message); QApplication.processEvents()

    def _progress(self,fraction):
        self.progress.setValue(int(max(0.0,min(1.0,fraction))*100)); QApplication.processEvents()
