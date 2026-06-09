# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create minimal state-store schema.

Revision ID: 0001_state_store
Revises:
Create Date: 2026-06-08
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_state_store"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "studies",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.PrimaryKeyConstraint("name"),
    )

    op.create_table(
        "study_admins",
        sa.Column("study_name", sa.String(length=255), nullable=False),
        sa.Column("user_name", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(["study_name"], ["studies.name"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("study_name", "user_name"),
    )
    op.create_index("idx_study_admins_user", "study_admins", ["user_name"])

    op.create_table(
        "study_orgs",
        sa.Column("study_name", sa.String(length=255), nullable=False),
        sa.Column("org", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(["study_name"], ["studies.name"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("study_name", "org"),
    )

    op.create_table(
        "study_sites",
        sa.Column("study_name", sa.String(length=255), nullable=False),
        sa.Column("site_name", sa.String(length=255), nullable=False),
        sa.Column("org", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(["study_name"], ["studies.name"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("study_name", "site_name"),
    )
    op.create_index("idx_study_sites_org", "study_sites", ["org"])

    op.create_table(
        "jobs",
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("study", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("content_uri", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=255), nullable=True),
        sa.Column("content_size", sa.BigInteger(), nullable=True),
        sa.Column("meta_json", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("job_id"),
    )
    op.create_index("idx_jobs_status", "jobs", ["status"])
    op.create_index("idx_jobs_study_status", "jobs", ["study", "status"])

    op.create_table(
        "submit_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("study_hash", sa.String(length=64), nullable=False),
        sa.Column("submitter_hash", sa.String(length=64), nullable=False),
        sa.Column("submit_token_hash", sa.String(length=64), nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("record_json", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("study_hash", "submitter_hash", "submit_token_hash", name="uq_submit_record_scope"),
    )
    op.create_index("idx_submit_records_job_id", "submit_records", ["job_id"])

    op.create_table(
        "disabled_clients",
        sa.Column("client_name", sa.String(length=255), nullable=False),
        sa.Column("disabled_by", sa.String(length=255), nullable=True),
        sa.Column("disabled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("client_name"),
    )

    op.create_table(
        "state_store_migrations",
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("source_format", sa.String(length=64), nullable=False),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("nvflare_version", sa.String(length=64), nullable=True),
        sa.Column("summary_json", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("name"),
    )


def downgrade():
    op.drop_table("state_store_migrations")
    op.drop_table("disabled_clients")
    op.drop_index("idx_submit_records_job_id", table_name="submit_records")
    op.drop_table("submit_records")
    op.drop_index("idx_jobs_study_status", table_name="jobs")
    op.drop_index("idx_jobs_status", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("idx_study_sites_org", table_name="study_sites")
    op.drop_table("study_sites")
    op.drop_table("study_orgs")
    op.drop_index("idx_study_admins_user", table_name="study_admins")
    op.drop_table("study_admins")
    op.drop_table("studies")
