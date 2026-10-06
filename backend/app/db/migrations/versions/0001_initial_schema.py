"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-10-06

"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("email", sa.String(), nullable=False, unique=True, index=True),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False, server_default="store_analyst"),
    )

    op.create_table(
        "stores",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("store_type", sa.String(), nullable=True),
        sa.Column("assortment", sa.String(), nullable=True),
        sa.Column("competition_distance", sa.Float(), nullable=True),
        sa.Column("cluster_id", sa.Integer(), nullable=True),
    )

    op.create_table(
        "clusters",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("store_count", sa.Integer(), server_default="0"),
    )

    op.create_table(
        "forecasts",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("store_id", sa.Integer(), sa.ForeignKey("stores.id"), nullable=False, index=True),
        sa.Column("forecast_date", sa.DateTime(), nullable=False),
        sa.Column("horizon_day", sa.Integer(), nullable=False),
        sa.Column("predicted_sales", sa.Float(), nullable=False),
        sa.Column("model_version", sa.Integer(), server_default="1"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "purchase_orders",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("store_id", sa.Integer(), sa.ForeignKey("stores.id"), nullable=False, index=True),
        sa.Column("recommended_qty", sa.Float(), nullable=False),
        sa.Column("current_inventory", sa.Float(), nullable=False),
        sa.Column("forecasted_demand", sa.Float(), nullable=False),
        sa.Column("status", sa.String(), server_default="pending"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "daily_store_features",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("store_id", sa.Integer(), nullable=False, index=True),
        sa.Column("date", sa.Date(), nullable=False, index=True),
        sa.Column("features", sa.JSON(), nullable=False),
    )
    op.create_index(
        "ix_daily_store_features_store_date",
        "daily_store_features",
        ["store_id", "date"],
        unique=True,
    )

    op.create_table(
        "inventory",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("store_id", sa.Integer(), nullable=False, unique=True, index=True),
        sa.Column("current_quantity", sa.Float(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("inventory")
    op.drop_index("ix_daily_store_features_store_date", table_name="daily_store_features")
    op.drop_table("daily_store_features")
    op.drop_table("purchase_orders")
    op.drop_table("forecasts")
    op.drop_table("clusters")
    op.drop_table("stores")
    op.drop_table("users")
