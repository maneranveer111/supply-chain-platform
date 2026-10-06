"""add cluster profile column

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-07

Adds a JSON `profile` column to the `clusters` table.
This column stores derived cluster-level forecast parameters
(mean_daily_sales, dow_multipliers, promo_uplift) so that
cold-start forecasting does not require requerying the training data.

The column is nullable because existing cluster rows will have
profile=NULL until the next cluster refresh task runs.
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("clusters", sa.Column("profile", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("clusters", "profile")
