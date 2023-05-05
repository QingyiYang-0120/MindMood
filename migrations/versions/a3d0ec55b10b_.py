"""empty message

Revision ID: a3d0ec55b10b
Revises: 0db0c43a1c47
Create Date: 2023-03-26 22:16:56.156067

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a3d0ec55b10b'
down_revision = '0db0c43a1c47'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('contactInfo', schema=None) as batch_op:
        batch_op.add_column(sa.Column('tele_num', sa.String(length=15), nullable=False))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('contactInfo', schema=None) as batch_op:
        batch_op.drop_column('tele_num')

    # ### end Alembic commands ###