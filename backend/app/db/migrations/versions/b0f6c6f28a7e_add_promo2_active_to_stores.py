"""add promo2_active to stores

Revision ID: b0f6c6f28a7e
Revises: 0003
Create Date: 2026-10-07 01:19:59.432118

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b0f6c6f28a7e'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('stores', sa.Column('promo2_active', sa.Boolean(), server_default='false', nullable=False))


def downgrade() -> None:
    op.drop_column('stores', 'promo2_active')
