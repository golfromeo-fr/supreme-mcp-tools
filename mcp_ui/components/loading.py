"""
Loading components for MCP UI.

Provides skeleton placeholders and spinners for non-blocking UI.
"""

from nicegui import ui

from ..logging_config import get_logger

logger = get_logger(__name__)


def loading_skeleton(lines: int = 3) -> ui.column:
    """
    Show placeholder skeleton while loading.

    Args:
        lines: Number of skeleton lines to display

    Returns:
        Column containing skeleton elements
    """
    logger.debug(f"component: loading_skeleton lines={lines}")

    with ui.column().classes("w-full gap-2") as container:
        for i in range(lines):
            # Vary width for visual interest
            width = "w-full" if i == 0 else f"w-{['full', '4/5', '3/4'][i % 3]}"
            with ui.row().classes(f"{width} h-6 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"):
                pass

    return container


def loading_spinner(message: str = "Loading...") -> ui.row:
    """
    Show spinner with message.

    Args:
        message: Loading message to display

    Returns:
        Row containing spinner and message
    """
    logger.debug(f"component: loading_spinner message='{message}'")

    with ui.row().classes("items-center gap-3 p-4") as container:
        ui.spinner(size="lg")
        ui.label(message).classes("text-gray-500")

    return container


def loading_overlay(message: str = "Loading...") -> ui.card:
    """
    Show loading overlay on top of content.

    Use this to show loading state without removing content.

    Args:
        message: Loading message to display

    Returns:
        Card overlay with spinner
    """
    logger.debug(f"component: loading_overlay message='{message}'")

    with ui.card().classes(
        "absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 z-50"
    ) as overlay:
        with ui.row().classes("items-center gap-3"):
            ui.spinner(size="lg")
            ui.label(message).classes("text-gray-700 dark:text-gray-300")

    return overlay
