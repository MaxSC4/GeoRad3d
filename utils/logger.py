import logging
from pathlib import Path

# Dossier par défaut pour les logs (créé s’il n’existe pas)
DEFAULT_LOG_DIR = Path("outputs/logs")
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Fichier de log principal
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "georad3d.log"


def setup_root_logger(level: int = logging.INFO, log_file: Path = DEFAULT_LOG_FILE) -> None:
    """
    Configure le logger racine (appelé automatiquement par get_logger).
    - Sortie console colorée (niveau, message)
    - Sortie fichier complète (timestamp, module, niveau, message)
    """
    if len(logging.getLogger().handlers) > 0:
        # Déjà configuré
        return

    log_format_console = "[%(levelname)s] %(message)s"
    log_format_file = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format_console))

    # Fichier handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format_file))

    # Root logger
    logging.basicConfig(level=level, handlers=[console_handler, file_handler])


def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Retourne un logger configuré pour le module demandé.
    Exemple :
        from utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Interpolation terminée")
    """
    setup_root_logger(level)
    logger = logging.getLogger(name or "georad3d")
    logger.setLevel(level)
    return logger


# Optionnel : exemple de test si exécuté seul
if __name__ == "__main__":
    log = get_logger("test_logger")
    log.debug("Message DEBUG (détails techniques)")
    log.info("Message INFO (normal)")
    log.warning("Message WARNING (anomalie non critique)")
    log.error("Message ERROR (erreur sérieuse)")
    print(f"✅ Logs écrits dans : {DEFAULT_LOG_FILE}")
