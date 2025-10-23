"""initial

Revision ID: 0001_initial
Revises: 
Create Date: 2025-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'model_meta',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_name', sa.String(), nullable=False, unique=True, index=True),
        sa.Column('image_type', sa.String(), nullable=True, index=True),
        sa.Column('minio_path', sa.String(), nullable=True),
        sa.Column('framework', sa.String(), nullable=True),
        sa.Column('file_name', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
    )

def downgrade():
    op.drop_table('model_meta')
