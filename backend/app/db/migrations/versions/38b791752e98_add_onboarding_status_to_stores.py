"""add_onboarding_status_to_stores

Revision ID: 38b791752e98
Revises: b0f6c6f28a7e
Create Date: 2026-10-07 10:56:28.916013

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '38b791752e98'
down_revision = 'b0f6c6f28a7e'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('stores', sa.Column('onboarding_status', sa.String(), server_default='ready', nullable=True))


def downgrade() -> None:
    op.drop_column('stores', 'onboarding_status')
