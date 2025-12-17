"""Initial database schema

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create routers table
    op.create_table(
        'routers',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_name', sa.String(255), nullable=False),
        sa.Column('manufacturer', sa.String(100), nullable=False),
        sa.Column('frequency_bands', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('max_tx_power_dbm', sa.Integer(), nullable=False),
        sa.Column('antenna_gain_dbi', sa.Numeric(4, 2), nullable=False),
        sa.Column('wifi_standard', sa.String(50), nullable=True),
        sa.Column('max_range_meters', sa.Integer(), nullable=True),
        sa.Column('coverage_area_sqm', sa.Integer(), nullable=True),
        sa.Column('price_usd', sa.Numeric(10, 2), nullable=True),
        sa.Column('specs', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id', name='pk_routers'),
        sa.CheckConstraint('max_tx_power_dbm BETWEEN 0 AND 30', name='valid_tx_power'),
        sa.CheckConstraint('antenna_gain_dbi BETWEEN -10 AND 20', name='valid_gain'),
    )
    op.create_index('ix_routers_model_name', 'routers', ['model_name'])
    op.create_index('ix_routers_manufacturer', 'routers', ['manufacturer'])
    op.create_index('ix_routers_wifi_standard', 'routers', ['wifi_standard'])
    op.create_index('ix_routers_price_usd', 'routers', ['price_usd'])

    # Create projects table
    op.create_table(
        'projects',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('map_image_path', sa.Text(), nullable=True),
        sa.Column('map_data', postgresql.JSONB(), nullable=False),
        sa.Column('scale_meters_per_pixel', sa.Numeric(10, 6), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id', name='pk_projects'),
    )
    op.create_index('ix_projects_user_id', 'projects', ['user_id'])

    # Create optimization_jobs table
    op.create_table(
        'optimization_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('constraints', postgresql.JSONB(), nullable=False),
        sa.Column('progress_percent', sa.Integer(), default=0),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id', name='pk_optimization_jobs'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE', name='fk_optimization_jobs_project_id_projects'),
        sa.CheckConstraint("status IN ('queued', 'running', 'completed', 'failed')", name='valid_status'),
        sa.CheckConstraint('progress_percent BETWEEN 0 AND 100', name='valid_progress'),
    )
    op.create_index('ix_optimization_jobs_project_id', 'optimization_jobs', ['project_id'])
    op.create_index('ix_optimization_jobs_status', 'optimization_jobs', ['status'])

    # Create solutions table
    op.create_table(
        'solutions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('router_placements', postgresql.JSONB(), nullable=False),
        sa.Column('coverage_percentage', sa.Numeric(5, 2), nullable=False),
        sa.Column('total_cost', sa.Numeric(10, 2), nullable=True),
        sa.Column('average_signal_strength', sa.Numeric(6, 2), nullable=True),
        sa.Column('min_signal_strength', sa.Numeric(6, 2), nullable=True),
        sa.Column('signal_heatmap_path', sa.Text(), nullable=True),
        sa.Column('metrics', postgresql.JSONB(), nullable=True),
        sa.Column('rank', sa.Integer(), default=1),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id', name='pk_solutions'),
        sa.ForeignKeyConstraint(['job_id'], ['optimization_jobs.id'], ondelete='CASCADE', name='fk_solutions_job_id_optimization_jobs'),
        sa.CheckConstraint('coverage_percentage BETWEEN 0 AND 100', name='valid_coverage'),
    )
    op.create_index('ix_solutions_job_id', 'solutions', ['job_id'])
    op.create_index('ix_solutions_rank', 'solutions', ['rank'])

    # Create feedback table
    op.create_table(
        'feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('solution_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=True),
        sa.Column('accuracy_score', sa.Numeric(3, 2), nullable=True),
        sa.Column('actual_measurements', postgresql.JSONB(), nullable=True),
        sa.Column('comments', sa.Text(), nullable=True),
        sa.Column('submitted_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id', name='pk_feedback'),
        sa.ForeignKeyConstraint(['solution_id'], ['solutions.id'], ondelete='CASCADE', name='fk_feedback_solution_id_solutions'),
        sa.CheckConstraint('rating BETWEEN 1 AND 5', name='valid_rating'),
        sa.CheckConstraint('accuracy_score BETWEEN 0 AND 1', name='valid_accuracy'),
    )
    op.create_index('ix_feedback_solution_id', 'feedback', ['solution_id'])


def downgrade() -> None:
    op.drop_table('feedback')
    op.drop_table('solutions')
    op.drop_table('optimization_jobs')
    op.drop_table('projects')
    op.drop_table('routers')
