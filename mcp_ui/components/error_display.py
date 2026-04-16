"""
Error display components for MCP UI.

Provides error and success notification components.
"""

import json
from pathlib import Path
from collections.abc import Callable

from nicegui import ui

from ..logging_config import get_logger

logger = get_logger(__name__)


def _load_ui_config() -> dict:
    """Load UI configuration from ports.json."""
    possible_paths = [
        Path(__file__).parent.parent / "config" / "ports.json",
        Path(__file__).parent.parent / "ports.json",
    ]
    for path in possible_paths:
        if path.exists():
            try:
                with Path(path).open() as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def _get_notification_durations() -> dict:
    """Get notification durations from config."""
    config = _load_ui_config()
    notifications = config.get("notifications", {})
    return {
        "success": notifications.get("success_duration", 3.0),
        "warning": notifications.get("warning_duration", 5.0),
        "info": notifications.get("info_duration", 3.0),
    }


_NOTIFICATION_DURATIONS = _get_notification_durations()


def show_error(
    message: str,
    on_dismiss: Callable[[], None] | None = None,
    on_retry: Callable[[], None] | None = None
) -> ui.card:
    """
    Show error message with dismiss and optional retry button.

    Args:
        message: Error message to display
        on_dismiss: Optional callback when dismissed
        on_retry: Optional callback when retry button clicked

    Returns:
        Card containing error display
    """
    logger.info(f"component: show_error message='{message[:50]}...'")

    def handle_dismiss() -> None:
        logger.debug(f"button_click: dismiss_error")
        if on_dismiss:
            on_dismiss()

    def handle_retry() -> None:
        logger.debug(f"button_click: retry")
        if on_retry:
            on_retry()

    with ui.card().classes("w-full bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800") as card:
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("error", color="red").classes("text-2xl")
                ui.label(message).classes("text-red-700 dark:text-red-300")

            with ui.row().classes("gap-1"):
                if on_retry:
                    ui.button(
                        icon="refresh",
                        on_click=handle_retry
                    ).props("flat dense round").classes("text-red-500")
                ui.button(
                    icon="close",
                    on_click=handle_dismiss
                ).props("flat dense round").classes("text-red-500")

    return card


def show_success(message: str, duration: float = None) -> None:
    """
    Show success notification that auto-dismisses.

    Args:
        message: Success message to display
        duration: Time in seconds before auto-dismiss (default: from config)
    """
    if duration is None:
        duration = _NOTIFICATION_DURATIONS["success"]
    logger.info(f"component: show_success message='{message[:50]}...' duration={duration}")

    ui.notify(
        message=message,
        type="positive",
        position="top",
        timeout=int(duration * 1000),
        close_button=True,
    )


def show_warning(message: str, duration: float = None) -> None:
    """
    Show warning notification that auto-dismisses.

    Args:
        message: Warning message to display
        duration: Time in seconds before auto-dismiss (default: from config)
    """
    if duration is None:
        duration = _NOTIFICATION_DURATIONS["warning"]
    logger.info(f"component: show_warning message='{message[:50]}...' duration={duration}")

    ui.notify(
        message=message,
        type="warning",
        position="top",
        timeout=int(duration * 1000),
        close_button=True,
    )


def show_info(message: str, duration: float = None) -> None:
    """
    Show info notification that auto-dismisses.

    Args:
        message: Info message to display
        duration: Time in seconds before auto-dismiss (default: from config)
    """
    if duration is None:
        duration = _NOTIFICATION_DURATIONS["info"]
    logger.info(f"component: show_info message='{message[:50]}...' duration={duration}")

    ui.notify(
        message=message,
        type="info",
        position="top",
        timeout=int(duration * 1000),
        close_button=True,
    )
