# src/utils/hooks.py
from __future__ import annotations
from typing import Callable, Optional
from pyannote.audio.pipelines.utils.hook import ProgressHook
from rich.console import Console
from rich.progress import (
    Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
)

class ForcedProgressHook(ProgressHook):
    """
    Extiende el ProgressHook oficial de pyannote.
    • new_phase(nombre, total)  → devuelve lambda advance(n)
    • close_phase()             → marca la tarea como concluida
    Todas las barras comparten el mismo estilo Rich.
    """

    def __init__(self, transient: bool = False) -> None:
        self.console = Console()
        super().__init__(transient=transient)
        self._task_id: Optional[int] = None   # id de la tarea Rich en curso

    # ------------- ciclo de vida del contexto ---------------------
    def __enter__(self) -> "ForcedProgressHook":
        # Creamos manualmente la barra con la consola compartida
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            transient=self.transient,
            console=self.console,          # ← la misma consola
        )
        self.progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # cierra la última fase si quedó abierta
        self.close_phase()
        self.progress.stop()
        super().__exit__(exc_type, exc_val, exc_tb)

    # ------------- API pública para el resto del pipeline ---------
    def new_phase(self, name: str, total: int = 1, unit: str = "") -> Callable[[int], None]:
        """
        Inicia una nueva tarea Rich y devuelve un atajo 'advance'.
        • name  : etiqueta que se muestra (se rellena a 15 caracteres para alinear)
        • total : número total de pasos; 1 si solo quieres un «tick» final
        • unit  : texto breve (p. ej. 'seg', 'batch'); opcional
        """
        self.close_phase()  # finaliza la tarea previa, si existe
        label = f"{name:15s}"
        self._task_id = self.progress.add_task(label, total=total, unit=unit)
        return lambda n=1: self.progress.update(self._task_id, advance=n)

    def close_phase(self) -> None:
        """Marca la fase actual como completada (si hay una en curso)."""
        if self._task_id is not None:
            task = self.progress.tasks[self._task_id]
            self.progress.update(self._task_id, completed=task.total)
            self._task_id = None
