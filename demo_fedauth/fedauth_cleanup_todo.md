# FedAuth Demo Cleanup TODO

Current cleanup backlog after the Keycloak/NVFlare demo work.

## Done

1. Block the most dangerous operator mistake in `prepare_startup_kits.sh`.
   - The script now checks for running FL demo containers before reprovisioning.
   - Goal: avoid wiping the mounted workspace under live server/client processes.

2. Make the demo docs explicit about the reprovisioning hazard.
   - `demo_fedauth/README.md` now says to stop the stack before rerunning startup-kit preparation.

## High Priority

1. Improve the reprovision workflow beyond the current guard.
   - Current fix: refuse reprovision while FL containers are up.
   - Better end-state: provision into a fresh workspace and switch over explicitly when desired.

2. Remove pid-file fragility in the container demo startup scripts.
   - Reprovisioning the mounted workspace currently breaks `pid.fl` / `daemon_pid.fl` assumptions.
   - The demo should restart cleanly without manual pid-file cleanup.

3. Add a containerized end-to-end smoke test for `demo_fedauth`.
   - Bring up Keycloak + server + 2 clients.
   - Log in with OIDC/token.
   - Run `check_status`, `list_jobs`, `submit_job`.
   - Fail CI if the demo path regresses.

4. Improve demo readiness checks.
   - Wait for Keycloak discovery endpoint.
   - Wait for server admin port readiness.
   - Wait for both clients to register before asking the operator to log in.

## Medium Priority

5. Reduce the rebuild time of the demo container image.
   - Current `Dockerfile` invalidates the dependency-install layer on most repo changes.
   - Split dependency install from source copy to improve cache reuse.

6. Publish OIDC metadata from the server to the admin CLI.
   - Keep only local callback/browser settings in the console profile.
   - Move issuer/client_id/audience/discovery defaults to a server-published auth metadata endpoint.

7. Remove the misleading admin console identity path.
   - The generated profile still lives under `admin@nvidia.com/`.
   - That should become a neutral console-profile name not tied to a fake human participant.

8. Make login errors more actionable in the CLI.
   - Show explicit token/OIDC failure reasons.
   - Avoid retry loops that hide the real root cause.

## Lower Priority

9. Consolidate fedauth demo documentation.
   - Keep `demo_fedauth/README.md` as the single operator guide.
   - Keep the slide deck and README aligned as the demo steps change.
