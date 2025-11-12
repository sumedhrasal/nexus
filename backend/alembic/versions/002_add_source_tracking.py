"""Add source tracking to entities

Revision ID: 002
Revises: 001
Create Date: 2025-11-10

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    # Add source tracking columns to entities table
    op.add_column('entities', sa.Column('source_type', sa.String(length=50), nullable=True))
    op.add_column('entities', sa.Column('source_id', sa.String(length=512), nullable=True))

    # Create index for source deduplication lookups
    op.create_index(
        'idx_source_dedup',
        'entities',
        ['collection_id', 'source_type', 'source_id'],
        unique=False  # Not unique because source_type/source_id can be NULL
    )


def downgrade():
    # Drop index
    op.drop_index('idx_source_dedup', table_name='entities')

    # Drop columns
    op.drop_column('entities', 'source_id')
    op.drop_column('entities', 'source_type')
