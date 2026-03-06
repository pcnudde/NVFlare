---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 26px;
  }
  h1, h2 {
    color: #0b5d3b;
  }
  table {
    font-size: 0.8em;
  }
  code {
    font-size: 0.82em;
  }
---

# FedAuth Demo

Keycloak-backed human auth for NVFlare, with site mTLS unchanged.

- Humans: browser OIDC login
- Sites: existing startup-kit PKI and mTLS
- Demo focus: secure admin login and job submission end to end
- Keycloak is configured from imported JSON
- `project.yml` contains only sites

---

## Fully Federated Auth Model

![w:1120](diagrams/federated_model.svg)

Central Keycloak brokers org-specific SSOs and emits the single token NVFlare trusts.

---

## Demo Topology

![w:1120](diagrams/topology.svg)

- Admin CLI runs on the host
- Keycloak, server, and clients run in containers
- Sites remain on mTLS; only human auth changes

---

## Browser Login Flow

![w:1120](diagrams/browser_login.svg)

- Standard authorization-code + PKCE browser login
- Server validates issuer, audience, and claims

---

## Job Submission Integrity

![w:1160](diagrams/job_submission.svg)

- no human signing key
- the server attests the accepted package and submitter identity
- both clients verify before run

---

# Appendix

Additional detail for discussion or Q&A.

---

## What Changes

| Concern | Sites | Humans (legacy) | Humans (demo) |
| --- | --- | --- | --- |
| Authentication | mTLS certs | mTLS certs | OIDC browser login |
| Identity source | cert CN + org | cert CN + org | IdP claims |
| Lifecycle | provisioned, long-lived | provisioned, long-lived | IdP-managed, dynamic |
| Startup kit | yes | yes | generated console profile only |

---

## Trust Split

![w:1040](diagrams/trust_split.svg)

This demo changes only the human plane.

---

## Tradeoff vs Legacy Admin Certs

- Legacy: human cert proved identity and signed the job
- Demo: OIDC proves identity and the server attests the accepted package
- Tradeoff: provenance now depends on the NVFlare control plane

---

## Future Option C: Keyless Signing With Sigstore

![w:1080](diagrams/option_c.svg)

- user signs the manifest with an ephemeral key
- an OIDC-backed signer issues a short-lived cert
- server verifies the bundle and countersigns acceptance
- executors verify both the user bundle and server countersignature

---

## Live Demo Script

1. Prepare startup kits
2. Start Keycloak, server, and two clients
3. Run `fl_admin.sh` and log in as `alice`
4. Run:

```text
check_status client
submit_job hello-numpy-sag
list_jobs
```

Main message: NVFlare can separate human auth from site auth without changing site mTLS.
