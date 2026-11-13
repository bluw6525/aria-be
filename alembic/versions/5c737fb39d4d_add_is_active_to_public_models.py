"""add_is_active_to_public_models

Revision ID: 5c737fb39d4d
Revises: 4b437c8f5ef0
Create Date: 2026-11-14 16:32:32.873729

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5c737fb39d4d'
down_revision: Union[str, Sequence[str], None] = '4b437c8f5ef0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [c['name'] for c in inspector.get_columns('public_models')]
    indexes = [idx['name'] for idx in inspector.get_indexes('public_models')]

    with op.batch_alter_table('public_models', schema=None) as batch_op:
        if 'is_active' not in columns:
            batch_op.add_column(sa.Column('is_active', sa.Boolean(), server_default='0', nullable=False))
        if 'ix_public_models_is_active' not in indexes:
            batch_op.create_index(batch_op.f('ix_public_models_is_active'), ['is_active'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('public_models', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_public_models_is_active'))
        batch_op.drop_column('is_active')
