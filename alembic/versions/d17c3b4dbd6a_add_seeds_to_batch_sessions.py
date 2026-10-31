"""add_seeds_to_batch_sessions

Revision ID: d17c3b4dbd6a
Revises: 820ad382148c
Create Date: 2026-11-01 06:30:18.142735

"""
import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd17c3b4dbd6a'
down_revision: Union[str, Sequence[str], None] = '820ad382148c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('batch_sessions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('seeds', sa.JSON(), nullable=True))

    # Backfill seeds untuk sesi yang sudah ada di database dari batch_iterations
    conn = op.get_bind()
    session_rows = conn.execute(sa.text("SELECT id FROM batch_sessions")).fetchall()
    for row in session_rows:
        s_id = row[0]
        it_rows = conn.execute(
            sa.text(
                "SELECT random_state FROM batch_iterations "
                "WHERE session_id = :s_id AND random_state IS NOT NULL "
                "ORDER BY id ASC"
            ),
            {"s_id": s_id}
        ).fetchall()

        seen = set()
        seeds_list = []
        for it in it_rows:
            rnd = it[0]
            if rnd is not None and rnd not in seen:
                seen.add(rnd)
                seeds_list.append(int(rnd))

        if seeds_list:
            conn.execute(
                sa.text(
                    "UPDATE batch_sessions SET seeds = :seeds WHERE id = :s_id"
                ),
                {"seeds": json.dumps(seeds_list), "s_id": s_id}
            )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('batch_sessions', schema=None) as batch_op:
        batch_op.drop_column('seeds')
