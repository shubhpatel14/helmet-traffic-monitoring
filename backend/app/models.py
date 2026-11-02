
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    number_plate: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        index=True,
    )

    helmet_detected: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
    )

    detection_source: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="mock",
    )

    review_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        index=True,
    )

    evidence_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    detector_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        default=None,
    )

    ocr_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        default=None,
    )

    recognition_status: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        default=None,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
