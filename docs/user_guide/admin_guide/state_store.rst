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

Migration Before Startup
========================

Production server startup requires the State Store schema and legacy migration marker to exist. Run the migration
once before the first startup after installing or upgrading a server package:

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
