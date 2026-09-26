"""initial schema: inspections, predictions, feedback, model versions

Revision ID: 0001
Revises:
Create Date: 2026-09-02
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "inspections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("image_path", sa.String(255), nullable=False),
        sa.Column("overlay_path", sa.String(255), nullable=True),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("image_width", sa.Integer(), nullable=False),
        sa.Column("image_height", sa.Integer(), nullable=False),
        sa.Column("image_sha256", sa.String(64), nullable=False),
    )
    op.create_index("ix_inspections_created_at", "inspections", ["created_at"])
    op.create_index("ix_inspections_image_sha256", "inspections", ["image_sha256"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "inspection_id",
            sa.Integer(),
            sa.ForeignKey("inspections.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("predicted_label", sa.String(64), nullable=False),
        sa.Column("predicted_index", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("confidence_tier", sa.String(8), nullable=False),
        sa.Column("probabilities", sa.JSON(), nullable=False),
        sa.Column("needs_review", sa.Boolean(), nullable=False),
        sa.Column("evidence_coverage", sa.Float(), nullable=False),
        sa.Column("evidence_bbox", sa.JSON(), nullable=True),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("severity_score", sa.Integer(), nullable=False),
        sa.Column("severity_components", sa.JSON(), nullable=False),
        sa.Column("recommended_action", sa.Text(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
    )
    op.create_index("ix_predictions_inspection_id", "predictions", ["inspection_id"])
    op.create_index("ix_predictions_model_version", "predictions", ["model_version"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "inspection_id",
            sa.Integer(),
            sa.ForeignKey("inspections.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_correct", sa.Boolean(), nullable=False),
        sa.Column("corrected_label", sa.String(64), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("inspector_name", sa.String(64), nullable=True),
        sa.Column("model_version_at_feedback", sa.String(32), nullable=False),
        sa.Column("review_status", sa.String(16), nullable=False),
        sa.Column("reviewed_by", sa.String(64), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("inspection_id", name="uq_feedback_inspection"),
    )
    op.create_index("ix_feedback_inspection_id", "feedback", ["inspection_id"])
    op.create_index("ix_feedback_review_status", "feedback", ["review_status"])

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("version", sa.String(32), nullable=False, unique=True),
        sa.Column("experiment", sa.String(64), nullable=False),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("arch", sa.String(64), nullable=False),
        sa.Column("classes", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("model_versions")
    op.drop_table("feedback")
    op.drop_table("predictions")
    op.drop_table("inspections")
