"""Pointmap conventions shared by training, diagnostics, and inference."""

POINTMAP_PREPROCESS_MODES = ("masked", "legacy_unmasked")


def is_legacy_unmasked(mode: str) -> bool:
    if mode not in POINTMAP_PREPROCESS_MODES:
        raise ValueError(
            f"Invalid pointmap_preprocess_mode={mode!r}; "
            f"expected one of {POINTMAP_PREPROCESS_MODES}"
        )
    return mode == "legacy_unmasked"
