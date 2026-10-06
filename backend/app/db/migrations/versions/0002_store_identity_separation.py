"""store identity separation and multi-tenant foundation

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-06

What this migration does
------------------------
1. Adds new columns to `stores`:
   - user_id         (nullable FK → users.id)      — multi-tenant ownership
   - name            (nullable VARCHAR)             — human-readable label
   - benchmark_store_id (nullable INTEGER, indexed) — ONLY link to Rossmann
                                                       training store (1..1115)
   - forecast_mode   (VARCHAR default 'auto')       — routing hint for Phase 2

2. Adds FK from stores.cluster_id → clusters.id
   Safe: clusters table has 0 rows; existing stores have cluster_id = NULL.

3. Adds FK from inventory.store_id → stores.id
   Safe: all 10 inventory rows reference store IDs 1–10, which all exist.

4. Adds FK from daily_store_features.store_id → stores.id
   Safe: all 390 rows reference store IDs 1–10, which all exist.

5. Creates new table `store_scalers` (per-application-store custom scaler)
   NOTE: this is NOT populated from saved_models/store_scalers.pkl.

6. Creates new table `store_sales` (raw daily sales for CSV upload pipeline)

Data-safety decisions for existing rows
----------------------------------------
- Existing 10 stores get user_id = NULL (they are demo/system stores,
  not owned by any specific user).  A future admin-controlled process
  can assign ownership.  We do NOT silently assign them to user id=1.
- Existing stores get benchmark_store_id = NULL.  Even though their
  database IDs (1–10) overlap with Rossmann store IDs (1–1115), this
  migration does NOT auto-map them.  benchmark_store_id must be set
  explicitly by an admin operation (Phase 2+).
- forecast_mode gets server_default='auto' so all existing rows receive
  the value 'auto' without a Python-level backfill.
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. Add new columns to `stores`
    # ------------------------------------------------------------------

    # user_id: nullable FK — existing stores become ownerless (user_id=NULL)
    op.add_column(
        "stores",
        sa.Column("user_id", sa.Integer(), nullable=True),
    )
    op.create_index("ix_stores_user_id", "stores", ["user_id"])
    op.create_foreign_key(
        "fk_stores_user_id_users",
        "stores", "users",
        ["user_id"], ["id"],
        ondelete="SET NULL",
    )

    # name: nullable human-readable store name
    op.add_column(
        "stores",
        sa.Column("name", sa.String(), nullable=True),
    )

    # benchmark_store_id: nullable integer, indexed, NO FK to another table.
    # This is an application-layer concept (keys into store_scalers.pkl).
    # Existing rows get NULL — they are NOT automatically mapped.
    op.add_column(
        "stores",
        sa.Column("benchmark_store_id", sa.Integer(), nullable=True),
    )
    op.create_index("ix_stores_benchmark_store_id", "stores", ["benchmark_store_id"])

    # forecast_mode: routing hint with safe server-side default
    op.add_column(
        "stores",
        sa.Column(
            "forecast_mode",
            sa.String(),
            nullable=False,
            server_default="auto",
        ),
    )

    # ------------------------------------------------------------------
    # 2. Add FK: stores.cluster_id → clusters.id
    #    Safe because clusters table has 0 rows and all cluster_id values
    #    in stores are currently NULL.
    # ------------------------------------------------------------------
    op.create_foreign_key(
        "fk_stores_cluster_id_clusters",
        "stores", "clusters",
        ["cluster_id"], ["id"],
        ondelete="SET NULL",
    )

    # ------------------------------------------------------------------
    # 3. Add FK: inventory.store_id → stores.id
    #    Safe: all 10 inventory rows have store_ids 1–10, which exist.
    # ------------------------------------------------------------------
    op.create_foreign_key(
        "fk_inventory_store_id_stores",
        "inventory", "stores",
        ["store_id"], ["id"],
        ondelete="CASCADE",
    )

    # ------------------------------------------------------------------
    # 4. Add FK: daily_store_features.store_id → stores.id
    #    Safe: all 390 feature rows have store_ids 1–10, which exist.
    # ------------------------------------------------------------------
    op.create_foreign_key(
        "fk_daily_store_features_store_id_stores",
        "daily_store_features", "stores",
        ["store_id"], ["id"],
        ondelete="CASCADE",
    )

    # ------------------------------------------------------------------
    # 5. Create `store_scalers` table
    #    One row per application store (UNIQUE on store_id).
    #    NOT populated from saved_models/store_scalers.pkl.
    # ------------------------------------------------------------------
    op.create_table(
        "store_scalers",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column(
            "store_id",
            sa.Integer(),
            sa.ForeignKey("stores.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("mean_log", sa.Float(), nullable=False),
        sa.Column("scale_log", sa.Float(), nullable=False),
        sa.Column("n_samples", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("store_id", name="uq_store_scalers_store_id"),
    )

    # ------------------------------------------------------------------
    # 6. Create `store_sales` table
    #    Raw daily sales rows per application store.
    #    Unique per (store_id, date) to prevent duplicates.
    # ------------------------------------------------------------------
    op.create_table(
        "store_sales",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column(
            "store_id",
            sa.Integer(),
            sa.ForeignKey("stores.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("date", sa.Date(), nullable=False, index=True),
        sa.Column("sales", sa.Float(), nullable=False),
        sa.Column("promo", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("school_holiday", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("state_holiday", sa.String(1), nullable=False, server_default="0"),
        sa.UniqueConstraint("store_id", "date", name="uq_store_sales_store_date"),
    )
    # Composite index for efficient per-store date-range queries
    op.create_index(
        "ix_store_sales_store_date",
        "store_sales",
        ["store_id", "date"],
    )


def downgrade() -> None:
    # Reverse order of upgrade

    # 6. Drop store_sales
    op.drop_index("ix_store_sales_store_date", table_name="store_sales")
    op.drop_table("store_sales")

    # 5. Drop store_scalers
    op.drop_table("store_scalers")

    # 4. Drop FK: daily_store_features.store_id → stores.id
    op.drop_constraint(
        "fk_daily_store_features_store_id_stores",
        "daily_store_features",
        type_="foreignkey",
    )

    # 3. Drop FK: inventory.store_id → stores.id
    op.drop_constraint(
        "fk_inventory_store_id_stores",
        "inventory",
        type_="foreignkey",
    )

    # 2. Drop FK: stores.cluster_id → clusters.id
    op.drop_constraint(
        "fk_stores_cluster_id_clusters",
        "stores",
        type_="foreignkey",
    )

    # 1. Remove new columns from stores (reverse of addition order)
    op.drop_column("stores", "forecast_mode")
    op.drop_index("ix_stores_benchmark_store_id", table_name="stores")
    op.drop_column("stores", "benchmark_store_id")
    op.drop_column("stores", "name")
    op.drop_constraint("fk_stores_user_id_users", "stores", type_="foreignkey")
    op.drop_index("ix_stores_user_id", table_name="stores")
    op.drop_column("stores", "user_id")
