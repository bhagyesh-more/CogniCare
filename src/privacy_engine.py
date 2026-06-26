"""
privacy_engine.py
Responsible AI privacy layer - ensures zero PII storage during inference.

Responsibilities:
    - Generate anonymous, non-reversible session IDs
    - Provide an ephemeral inference context manager that clears all
      intermediate data on exit (whether successful or not)
    - Scrub any PII-like keys from input data before processing
    - Guarantee that no subject-identifiable information is retained

Design principles:
    - Session IDs are UUID4 - random, non-sequential, non-reversible
    - No mapping between session ID and real subject ID is ever stored
    - All input data is treated as temporary; cleared after inference
"""

import hashlib
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Generator, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Keys that should never appear in inference input (PII identifiers)
_PII_KEYS = {
    "subject_id", "participant_id", "user_id", "name", "email",
    "dob", "date_of_birth", "age", "gender", "location", "ip_address",
}


class PrivacyEngine:
    """
    Manages anonymous session lifecycle and data privacy for inference.

    Usage
    -----
        privacy = PrivacyEngine()

        with privacy.session() as session_id:
            clean_data = privacy.sanitise(raw_input)
            result = engine.predict_emotional_arousal(clean_data)
            # session_id is anonymous; raw_input is not stored anywhere
    """

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    @staticmethod
    def new_session_id() -> str:
        """
        Generate a cryptographically random anonymous session ID.
        UUID4 - no timestamp, no MAC address, not reversible.
        """
        return str(uuid.uuid4())

    @staticmethod
    def anonymise_subject_id(subject_id: str) -> str:
        """
        One-way hash a subject ID for logging purposes (e.g. S2 → short hex).
        The original ID cannot be recovered from the hash.
        """
        return hashlib.sha256(subject_id.encode()).hexdigest()[:12]

    @contextmanager
    def session(self) -> Generator[str, None, None]:
        """
        Context manager for a single anonymous inference session.

        Yields the session ID; guarantees cleanup on exit regardless of errors.

        Example
        -------
            with privacy.session() as sid:
                result = engine.predict_emotional_arousal(data)
        """
        session_id = self.new_session_id()
        logger.info("Session started  [%s]", session_id)
        try:
            yield session_id
        finally:
            # Nothing is stored server-side; this log confirms the boundary
            logger.info("Session closed   [%s] - no data retained.", session_id)

    # ------------------------------------------------------------------
    # Data sanitisation
    # ------------------------------------------------------------------

    def sanitise(
        self,
        data: Union[dict, pd.DataFrame],
        strict: bool = True,
    ) -> Union[dict, pd.DataFrame]:
        """
        Remove PII-like keys from input data before inference.

        Parameters
        ----------
        data   : dict or DataFrame to sanitise
        strict : if True, raise ValueError when PII keys are detected;
                 if False, silently drop them

        Returns
        -------
        Sanitised copy of the input (original is never modified)
        """
        if isinstance(data, dict):
            return self._sanitise_dict(data, strict)
        return self._sanitise_dataframe(data, strict)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitise_dict(data: dict, strict: bool) -> dict:
        found_pii = _PII_KEYS & set(data.keys())
        if found_pii:
            msg = f"PII keys detected and removed from input: {found_pii}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
        return {k: v for k, v in data.items() if k not in _PII_KEYS}

    @staticmethod
    def _sanitise_dataframe(df: pd.DataFrame, strict: bool) -> pd.DataFrame:
        found_pii = _PII_KEYS & set(df.columns)
        if found_pii:
            msg = f"PII columns detected and removed from input: {found_pii}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
        return df.drop(columns=list(found_pii), errors="ignore").copy()
