from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWizard, QWizardPage)

from prompt_master.core.paths import AppPaths
from prompt_master.inference.device_detection import (QUANTIZATIONS, describe, detect_cpu,
    detect_devices, recommended_quantization, vram_shortfall_mb)
from prompt_master.provisioning import importer, installer, verifier

# Said on the model page when the model will be kept in system RAM. A
# disclaimer, not a warning: neither mode is sized against a memory figure, so
# there is no threshold here to be under and nothing to caution about.
CPU_NOTE = ("This install runs on the processor and system RAM. No NVIDIA GPU or driver is "
            "used, and none is required.")
MIXED_NOTE = ("This install loads the model into system RAM and uses {name} for the work "
              "llama.cpp can hand it — prompt processing and image encoding — so only a small "
              "amount of VRAM is held.")


class SetupWizard(QWizard):
    """Provision and validate a complete local runtime; no state is saved early.

    The questions asked here — directory, device, quantization, and whether the
    model is already on this machine — are the same ones the console installer
    asks, and both hand off to ``provisioning.installer`` for everything after
    the last answer, so the two front ends cannot provision differently.
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
        page=QWizardPage(); page.setTitle("What should run the model"); layout=QVBoxLayout(page)
        self.hardware_status=QLabel("Open this page to scan with nvidia-smi."); self.hardware_status.setWordWrap(True)
        self.gpu=QComboBox(); layout.addWidget(self.hardware_status); layout.addWidget(self.gpu); self.addPage(page)

    def _model_page(self):
        page=QWizardPage(); page.setTitle("Model quality"); form=QFormLayout(page)
        self.quant=QComboBox(); self.quant.addItems(list(QUANTIZATIONS)); self.recommendation=QLabel(); self.recommendation.setWordWrap(True)
        form.addRow("GGUF quantization",self.quant); form.addRow(self.recommendation)
        # A model already on this machine: 16-27 GiB not downloaded again, and
        # the only way past a connection that cannot carry that file.
        self.model_file=QLineEdit(); self.model_file.setPlaceholderText("Leave empty to download")
        browse=QPushButton("Browse…"); browse.clicked.connect(self._browse_model)
        row=QHBoxLayout(); row.addWidget(self.model_file); row.addWidget(browse)
        form.addRow("Model file you already have",row)
        form.addRow(QLabel("It is moved into the installation directory under its own name — "
                           "not copied, and not renamed."))
        self.quant.currentTextChanged.connect(self._describe_quant); self.addPage(page)

    def _browse_model(self):
        value,_=QFileDialog.getOpenFileName(self,"Model file",str(self.paths.root),"GGUF models (*.gguf)")
        if value: self.model_file.setText(value)

    def _install_page(self):
        page=QWizardPage(); page.setTitle("Download, verify, and validate"); layout=QVBoxLayout(page)
        text=QLabel("Finish downloads the pinned llama.cpp runtime, model, and matching vision projector; verifies size and SHA-256; then validates one text and one image request."); text.setWordWrap(True); layout.addWidget(text)
        self.progress=QProgressBar(); self.install_status=QLabel("Ready"); self.install_status.setWordWrap(True); layout.addWidget(self.progress); layout.addWidget(self.install_status); self.addPage(page)

    def _browse(self):
        value=QFileDialog.getExistingDirectory(self,"Installation directory",self.location.text())
        if value: self.location.setText(value)

    def _page_changed(self,index):
        if index == 1:
            # The processor is always in the list and always last, so a machine
            # with no NVIDIA driver still has something to select. A scan that
            # fails outright leaves it as the only entry rather than an empty
            # page, which is exactly what such a machine would have chosen.
            self.gpu.clear()
            try: self.gpus=detect_devices()
            except Exception as exc: self.gpus=[detect_cpu()]; self.hardware_status.setText(f"GPU scan failed: {exc}\nThe processor is still available.")
            else: self.hardware_status.setText(self._hardware_summary())
            for gpu in self.gpus: self.gpu.addItem(self._device_label(gpu),gpu)
        elif index == 2 and self.gpu.currentData():
            self.quant.setCurrentText(recommended_quantization(self.gpu.currentData())); self._describe_quant()

    def _hardware_summary(self):
        cards=[gpu for gpu in self.gpus if not (gpu.is_cpu or gpu.is_mixed)]
        if cards: return (f"Found {len(cards)} CUDA GPU(s), each offered two ways: holding the model in "
                          "its own memory, or in mixed mode, where the model is loaded into system RAM "
                          "and the card is used for the work llama.cpp can hand it. The processor is "
                          "offered too.")
        return "No CUDA GPU found. This install will run on the processor and system RAM."

    @staticmethod
    def _device_label(device):
        # One wording, shared with the runtime menu — see device_detection.describe.
        return describe(device)

    def _describe_quant(self,*_):
        """Recommendation, download size and the VRAM warning, together.

        The warning is shown rather than enforced: a card below the threshold
        still installs and runs, it just spills layers to system RAM, and that
        trade is the user's to make — and mixed mode is the other answer to it.
        With the weights in system RAM there is no threshold to be below, so
        what is shown there is a disclaimer and not a warning."""
        gpu=self.gpu.currentData()
        if gpu is None: return
        quant=self.quant.currentText()
        if gpu.weights_in_system_ram:
            # The weights are not in VRAM, so there is nothing to measure a
            # shortfall against: the download is the only thing that differs
            # between the three here.
            note=CPU_NOTE if gpu.is_cpu else MIXED_NOTE.format(name=gpu.name)
            where="" if gpu.is_cpu else " in mixed mode"
            self.recommendation.setText("\n".join([
                f"Recommended for {gpu.name}{where}: {recommended_quantization(gpu)}",
                f"Download: {installer.format_download_size(gpu,quant)}.", note]))
            return
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
            QMessageBox.critical(self,"Nothing selected","Select a CUDA GPU, or the processor."); return False
        elif self.currentId() == 2:
            try: self.sources=self._vet_model_file()
            except Exception as exc: QMessageBox.critical(self,"Model file",str(exc)); return False
            if self.sources is None: return False
        elif self.currentId() == 3 and not self.completed:
            try: self._provision()
            except Exception as exc: self.install_status.setText(f"Setup failed: {exc}"); QMessageBox.critical(self,"Setup failed",str(exc)); return False
        return super().validateCurrentPage()

    def _vet_model_file(self):
        """The supplied file's place in the manifest, decided before anything is
        downloaded or moved. ``None`` means the user went back to change it."""
        given=self.model_file.text().strip().strip('"')
        if not given: return {}
        path=Path(given).expanduser()
        key=f"model-{self.quant.currentText()}"
        component=installer.load_components()[key]
        problem=importer.size_problem(component,path)
        if problem is None:
            self._status(f"Checking {path.name} against the pinned SHA-256…")
            digest=verifier.digest_of(path,lambda done,total:self._progress(done/total))
            self._progress(0.0)
            if digest.casefold()==component.sha256.casefold():
                return {key:importer.LocalSource(path,checked=True)}
            identified=installer.identify(digest)
            problem=(f"{path.name} does not match {key} — "
                     + (f"it is {identified}." if identified else "it is not a file this build pins."))
        answer=QMessageBox.question(self,"Model file",f"{problem}\n\nUse it anyway?",
                                    QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer != QMessageBox.Yes: return None
        return {key:importer.LocalSource(path,checked=True)}

    def _provision(self):
        installer.provision(self.paths,self.gpu.currentData(),self.quant.currentText(),
                            sources=getattr(self,"sources",None),
                            on_status=self._status,on_progress=self._progress)
        self.progress.setValue(100); self.completed=True

    def _status(self,message):
        self.install_status.setText(message); QApplication.processEvents()

    def _progress(self,fraction):
        self.progress.setValue(int(max(0.0,min(1.0,fraction))*100)); QApplication.processEvents()
