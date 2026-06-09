.. _state_store:

State Store
*************

Overview
========

The State Store is the server-side database for durable server state. It stores records that must survive
server restart and can be shared by a future highly available server deployment:

- study definitions and study-user membership
- job metadata and job status
- submit-token records used for idempotent job submission
- disabled-client records
- migration markers

Large job bytes are not stored in the State Store. Job packages, workspace archives, logs, checkpoints, and other
large artifacts remain in the configured job storage or artifact storage.

Database Configuration
======================

The server configures the State Store as a component in ``local/resources.json`` or
``local/resources.json.default``.

For local development and simulation, SQLite can be used:

.. code-block:: json

   {
     "id": "state_store",
     "path": "nvflare.app_common.state_store.sql_store.SqlStateStore",
     "args": {
       "db_url": "sqlite:////path/to/server/state-store.db"
     }
   }

SQLite is not intended as the production HA database. If SQLite is used outside local development, the SQLite file must
be on durable storage.

A relative SQLite path (the provisioned default is ``sqlite:///state-store.db``) is resolved against the server
workspace directory, never against the process working directory. The server and the ``nvflare-state-store-migrate``
command (which resolves against ``--server-root``) therefore always open the same database file.

For production, configure an external database URL through an environment variable so credentials are not written into
``resources.json``:

.. code-block:: json

   {
     "id": "state_store",
     "path": "nvflare.app_common.state_store.sql_store.SqlStateStore",
     "args": {
       "db_url_env": "NVFLARE_STATE_STORE_DB_URL"
     }
   }

For PostgreSQL, install NVFLARE with the PostgreSQL State Store extra and set the environment variable before running
the migration or starting the server:

.. code-block:: shell

   export NVFLARE_STATE_STORE_DB_URL='postgresql+psycopg://user:password@db.example.com:5432/nvflare?sslmode=require'

If ``db_url_env`` is configured and the environment variable is missing, server startup fails closed.

Fresh Installs Bootstrap Automatically
======================================

Server startup requires the State Store schema and a migration marker to exist. On a fresh install — a server
workspace with no legacy filesystem state (no existing jobs, no ``local/study_registry.json``, no
``disabled_clients.json``) — the server bootstraps itself at startup: it applies the schema migrations and writes a
fresh-install marker. No manual step is needed for new POC, Docker, bare-metal, or Kubernetes deployments.

A freshly provisioned server kit whose ``project.yml`` defines studies contains a provision-time
``local/study_registry.json``. That file alone is configuration, not legacy runtime data: at first startup the server
imports its studies into the State Store automatically and writes the migration marker, so such kits also start with
just ``start.sh``. Only when the registry is accompanied by legacy runtime data (existing jobs or
``disabled_clients.json``) does startup fail closed and require the explicit migration described below.

If the ``state_store`` component is missing from ``resources.json`` (for example, a workspace provisioned by an older
NVFLARE release that was upgraded in place), the server falls back to a default SQLite database at
``<workspace>/state-store.db`` and logs a warning. Add an explicit ``state_store`` component to make the database
location deliberate.

Migration When Upgrading a Workspace with Legacy State
======================================================

If the workspace contains legacy filesystem state, startup fails closed instead of importing it implicitly. Run the
migration once before the first startup after upgrading such a server package:

.. code-block:: shell

   nvflare-state-store-migrate --server-root /path/to/server

The migration command:

- applies Alembic schema migrations to the configured database
- imports legacy filesystem job metadata
- imports ``local/study_registry.json`` if present
- imports ``disabled_clients.json`` if present
- writes a migration marker so later server startups know the legacy import has already happened

The command is idempotent after the marker is written. If the marker is missing but State Store data already exists in
the database, the migration fails instead of guessing whether import is safe.

The migration command is only needed for workspaces with legacy filesystem data. Fresh installs bootstrap themselves
at startup as described above.

If the workspace has no ``state_store`` component in ``resources.json`` and ``--db-url`` is not given, the command
falls back to the same default SQLite database the server uses (``<server-root>/state-store.db``) and prints a notice,
so legacy workspaces provisioned before the State Store existed can still be migrated.

If the State Store database is lost and re-created by re-running the migration, job metadata is rebuilt from the meta
stored alongside each job in job storage. Job status changes are mirrored to that storage meta on a best-effort basis
(mirror write failures are logged; the State Store remains authoritative while it exists), so a re-migrated database
reflects the last successfully mirrored status of each job. A job whose mirror writes failed can reappear with a stale
status (for example, a finished job resurfacing as SUBMITTED).

Kubernetes and HA Notes
=======================

For Kubernetes production deployments, use an external PostgreSQL database and inject the DB URL through a Secret-backed
environment variable. ``nvflare deploy prepare`` adds a server init container that runs the State Store migration
before the main server container starts.

For example, a Kubernetes deploy config can pass the DB URL Secret to both the migration init container and the server
container and configure the generated server resources to read that environment variable:

.. code-block:: yaml

   parent:
     docker_image: registry.example.com/nvflare:latest
     state_store:
       db_url_env: NVFLARE_STATE_STORE_DB_URL
     env:
       - name: NVFLARE_STATE_STORE_DB_URL
         valueFrom:
           secretKeyRef:
             name: state-store-db
             key: db-url

The init container is idempotent after the migration marker is written. The main server still validates the marker at
startup and fails closed if migration did not complete.

The server workspace can be ephemeral for runtime scratch data if job artifacts, logs, and other large outputs are stored
in durable artifact/job storage. A permanent PVC is only needed for production if you choose a filesystem-backed durable
store such as SQLite or local filesystem artifact storage.

Job runner subprocesses do not need direct State Store access. The parent server process owns State Store access and
passes per-job runtime state to child processes.

Operational Responsibilities
============================

Administrators should back up the State Store database together with the artifact/job storage that holds large bytes.
Restoring only one side is incomplete: the database contains job metadata and submit-token records, while artifact/job
storage contains the job package and other byte payloads.

Disabled-client enforcement reads the State Store through a short-TTL cache. If a store read fails and a cached value
exists for the client, the last cached value is used. If no cached value exists, the server **fails open** by default
(the client is treated as not disabled) so heartbeats and registrations keep working through a database blip; this
trades security for availability, since a disabled client never before seen by this server process could be admitted
while the database is unreachable. To fail closed instead, set ``NVFL_DISABLED_CLIENT_FAIL_CLOSED=1`` in the server
environment: such clients are treated as disabled until the database is reachable again. (Code that constructs
``ClientManager`` directly can pass ``disabled_check_fail_open=False`` as the programmatic override; an explicit
constructor argument takes precedence over the environment variable.) Only clients with no cached value on the server
are affected by this choice.
