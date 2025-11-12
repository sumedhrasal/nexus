"""Add search feedback and quality metrics

Revision ID: 001
Revises:
Create Date: 2025-11-10

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create search_feedback table
    op.create_table(
        'search_feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('search_analytics_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('result_position', sa.Integer(), nullable=False),
        sa.Column('result_entity_id', sa.String(length=512), nullable=False),
        sa.Column('feedback_type', sa.String(length=50), nullable=False),
        sa.Column('feedback_metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['search_analytics_id'], ['search_analytics.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create index on search_analytics_id for faster lookups
    op.create_index('idx_feedback_search_analytics', 'search_feedback', ['search_analytics_id'])


def downgrade():
    # Drop index
    op.drop_index('idx_feedback_search_analytics', table_name='search_feedback')

    # Drop table
    op.drop_table('search_feedback')
