"""add_performance_indexes

Revision ID: 68c71bd7b578
Revises: 38b791752e98
Create Date: 2026-10-08 15:35:49.610185

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '68c71bd7b578'
down_revision = '38b791752e98'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(op.f('ix_stores_onboarding_status'), 'stores', ['onboarding_status'], unique=False)
    op.create_index(op.f('ix_purchase_orders_status'), 'purchase_orders', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_purchase_orders_status'), table_name='purchase_orders')
    op.drop_index(op.f('ix_stores_onboarding_status'), table_name='stores')

