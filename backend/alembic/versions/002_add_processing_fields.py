"""Add processing status fields to projects

Revision ID: 002
Revises: 001
Create Date: 2025-12-16
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add processing status fields to projects table
    op.add_column('projects', sa.Column('processing_status', sa.String(50), nullable=False, server_default='pending'))
    op.add_column('projects', sa.Column('processing_progress', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('projects', sa.Column('detected_scale', sa.Float(), nullable=True))
    op.add_column('projects', sa.Column('scale_confidence', sa.Float(), nullable=True))
    op.add_column('projects', sa.Column('processing_error', sa.Text(), nullable=True))


def downgrade() -> None:
    # Remove processing status fields
    op.drop_column('projects', 'processing_error')
    op.drop_column('projects', 'scale_confidence')
    op.drop_column('projects', 'detected_scale')
    op.drop_column('projects', 'processing_progress')
    op.drop_column('projects', 'processing_status')
