# Slide 1

## Ephemeral Keys for NVFlare

### From FLARE-as-CA to external CA with ephemeral endpoint keys

High-level design review — draft 2026-07-02

Companion doc: `docs/design/ephemeral-key-architecture.md`

---

# Slide 2 — System components

```mermaid
flowchart LR
    AL[Admin laptop<br/>provisioning]
    DS[Data-science laptop<br/>admin CLI]

    subgraph SRV[Server site]
        SP[Server parent<br/>SP]
        SJ[Server job<br/>SJ]
    end

    subgraph CLT[Client site]
        CP[Client parent<br/>CP]
        CJ[Client job<br/>CJ]
    end

    AL -. startup kits .-> SP
    AL -. startup kits .-> CP
    DS -- admin session mTLS --> SP
    CP <-- register / FL mTLS --> SP
    SP -- spawn --> SJ
    CP -- spawn --> CJ
    SJ <-. internal listener .-> SP
    CJ <-. internal listener .-> CP
    CJ <-. FL tasks and results, routed via parents .-> SJ
```

---

# Slide 3 — Feature 1: SSO ephemeral keys for admins

```mermaid
flowchart LR
    F1[/"1. SSO ephemeral keys for admins"/]:::callout

    AL[Admin laptop<br/>provisioning]
    DS[Data-science laptop<br/>admin CLI]

    subgraph SRV[Server site]
        SP[Server parent<br/>SP]
        SJ[Server job<br/>SJ]
    end

    subgraph CLT[Client site]
        CP[Client parent<br/>CP]
        CJ[Client job<br/>CJ]
    end

    NOTE["• login via SSO / OIDC, step-ca issues cert<br/>• short-lived cert ~24 h, key stays on laptop<br/>• no admin startup kits to distribute or revoke<br/>• study entitlements can ride in the cert<br/>• status: PR #4846 open"]:::note

    F1 -.- DS
    AL -. startup kits .-> SP
    AL -. startup kits .-> CP
    DS -- admin session mTLS --> SP
    CP <-- register / FL mTLS --> SP
    SP -- spawn --> SJ
    CP -- spawn --> CJ
    SJ <-. internal listener .-> SP
    CJ <-. internal listener .-> CP
    CJ <-. FL tasks and results, routed via parents .-> SJ
    NOTE ~~~ CLT

    classDef callout fill:#ffe08a,stroke:#c98a00,stroke-width:2px
    classDef note fill:#eef6ff,stroke:#5b8def,text-align:left
```

---

# Slide 4 — Feature 2: ephemeral keys for server/client

```mermaid
flowchart LR
    F2[/"2. Ephemeral keys for server / client"/]:::callout

    AL[Admin laptop<br/>provisioning]
    DS[Data-science laptop<br/>admin CLI]

    subgraph SRV[Server site]
        SP[Server parent<br/>SP]
        SJ[Server job<br/>SJ]
    end

    subgraph CLT[Client site]
        CP[Client parent<br/>CP]
        CJ[Client job<br/>CJ]
    end

    NOTE["• enrollment agent + machine identity, CSR to step-ca<br/>• keys generated on the endpoint, never transported<br/>• 30-day certs (per-project override), renewed live, no restart<br/>• eviction = stop certifying, dies within one cert lifetime<br/>• builds on feat/external-workload-certs + renewal spike"]:::note

    F2 -.- SP
    F2 -.- CP
    AL -. startup kits .-> SP
    AL -. startup kits .-> CP
    DS -- admin session mTLS --> SP
    CP <-- register / FL mTLS --> SP
    SP -- spawn --> SJ
    CP -- spawn --> CJ
    SJ <-. internal listener .-> SP
    CJ <-. internal listener .-> CP
    CJ <-. FL tasks and results, routed via parents .-> SJ
    NOTE ~~~ CLT

    classDef callout fill:#ffe08a,stroke:#c98a00,stroke-width:2px
    classDef note fill:#eef6ff,stroke:#5b8def,text-align:left
```

---

# Slide 5 — Feature 3: ephemeral keys for jobs

```mermaid
flowchart LR
    F3[/"3. Ephemeral keys for jobs"/]:::callout

    AL[Admin laptop<br/>provisioning]
    DS[Data-science laptop<br/>admin CLI]

    subgraph SRV[Server site]
        SP[Server parent<br/>SP]
        SJ[Server job<br/>SJ]
    end

    subgraph CLT[Client site]
        CP[Client parent<br/>CP]
        CJ[Client job<br/>CJ]
    end

    NOTE["• SJ / CJ get their own key + cert per job<br/>• job-id claim in the cert; CN stays the site name<br/>• prototyped: root-signed job CA in server kit, SP issues at deploy<br/>• key injected via the launch channel, no enrollment inside the job<br/>• retires the launch-time auth token (-t / -ts args): today the CJ<br/>&nbsp;&nbsp;is handed the site's own registration token — a non-expiring<br/>&nbsp;&nbsp;bearer exposed via ps, pod specs, and on-disk launch files"]:::note

    F3 -.- SJ
    F3 -.- CJ
    AL -. startup kits .-> SP
    AL -. startup kits .-> CP
    DS -- admin session mTLS --> SP
    CP <-- register / FL mTLS --> SP
    SP -- spawn --> SJ
    CP -- spawn --> CJ
    SJ <-. internal listener .-> SP
    CJ <-. internal listener .-> CP
    CJ <-. FL tasks and results, routed via parents .-> SJ
    NOTE ~~~ AL

    classDef callout fill:#ffe08a,stroke:#c98a00,stroke-width:2px
    classDef note fill:#eef6ff,stroke:#5b8def,text-align:left
```

---

# Slide 6 — Feature 4: leverage KMS for provisioning

```mermaid
flowchart LR
    F4[/"4. Leverage KMS for provisioning"/]:::callout

    AL[Admin laptop<br/>provisioning]
    DS[Data-science laptop<br/>admin CLI]

    subgraph SRV[Server site]
        SP[Server parent<br/>SP]
        SJ[Server job<br/>SJ]
    end

    subgraph CLT[Client site]
        CP[Client parent<br/>CP]
        CJ[Client job<br/>CJ]
    end

    NOTE["• root key non-exportable in KMS / HSM, signs only intermediates<br/>• provision takes rootCA.pem as input: trust-only provisioning<br/>• no CA or participant private key in provisioning state<br/>• kits carry trust + identity metadata, not credentials<br/>• builds on the BYO-root seam; baseline stays for POCs"]:::note

    F4 -.- AL
    AL -. startup kits .-> SP
    AL -. startup kits .-> CP
    DS -- admin session mTLS --> SP
    CP <-- register / FL mTLS --> SP
    SP -- spawn --> SJ
    CP -- spawn --> CJ
    SJ <-. internal listener .-> SP
    CJ <-. internal listener .-> CP
    CJ <-. FL tasks and results, routed via parents .-> SJ
    NOTE ~~~ CLT

    classDef callout fill:#ffe08a,stroke:#c98a00,stroke-width:2px
    classDef note fill:#eef6ff,stroke:#5b8def,text-align:left
```

---

# Slide 7 — Feature 5: key rotation for server/client

```mermaid
flowchart LR
    F5[/"5. Key rotation for server / client"/]:::callout

    AL[Admin laptop<br/>provisioning]
    DS[Data-science laptop<br/>admin CLI]

    subgraph SRV[Server site]
        SP[Server parent<br/>SP]
        SJ[Server job<br/>SJ]
    end

    subgraph CLT[Client site]
        CP[Client parent<br/>CP]
        CJ[Client job<br/>CJ]
    end

    NOTE["• split: certs renew live, keys rotate at controlled restarts<br/>• keys bounded at 1 year or less (decided)<br/>• per-participant, uncoordinated; client restart = FL churn<br/>• server without a window: blue-green drain or<br/>&nbsp;&nbsp;job-surviving restart (decision D3)<br/>• no live key-swap machinery in the base posture"]:::note

    F5 -.- SP
    F5 -.- CP
    AL -. startup kits .-> SP
    AL -. startup kits .-> CP
    DS -- admin session mTLS --> SP
    CP <-- register / FL mTLS --> SP
    SP -- spawn --> SJ
    CP -- spawn --> CJ
    SJ <-. internal listener .-> SP
    CJ <-. internal listener .-> CP
    CJ <-. FL tasks and results, routed via parents .-> SJ
    NOTE ~~~ CLT

    classDef callout fill:#ffe08a,stroke:#c98a00,stroke-width:2px
    classDef note fill:#eef6ff,stroke:#5b8def,text-align:left
```

---

# Slide 8 — Feature 6: study entitlements in admin certs

```mermaid
flowchart LR
    F6[/"6. Study entitlements in admin certs"/]:::callout

    AL[Admin laptop<br/>provisioning]
    DS[Data-science laptop<br/>admin CLI]

    subgraph SRV[Server site]
        SP[Server parent<br/>SP]
        SJ[Server job<br/>SJ]
    end

    subgraph CLT[Client site]
        CP[Client parent<br/>CP]
        CJ[Client job<br/>CJ]
    end

    NOTE["• study entitlements ride in a signed X.509 extension<br/>&nbsp;&nbsp;of the SSO-issued admin cert<br/>• OR'd with the registry admins list — purely additive<br/>• registry keeps study existence, site enrollment, scheduling<br/>• malformed claim = reject login, never silently ignore<br/>• per-admin adoptable; static + SSO admins coexist"]:::note

    F6 -.- DS
    F6 -.- SP
    AL -. startup kits .-> SP
    AL -. startup kits .-> CP
    DS -- admin session mTLS --> SP
    CP <-- register / FL mTLS --> SP
    SP -- spawn --> SJ
    CP -- spawn --> CJ
    SJ <-. internal listener .-> SP
    CJ <-. internal listener .-> CP
    CJ <-. FL tasks and results, routed via parents .-> SJ
    NOTE ~~~ CLT

    classDef callout fill:#ffe08a,stroke:#c98a00,stroke-width:2px
    classDef note fill:#eef6ff,stroke:#5b8def,text-align:left
```

---

# Slide 9 — Summary

**The target is not a new mode — it is today's baseline (2.8) after
enabling six independent features, each valuable alone, several already
shipping.** The runtime is posture-blind: every process just reads
credential files and validates against rootCA.

| # | Feature | What it removes | Status | Open decisions |
|---|---------|-----------------|--------|----------------|
| 1 | SSO ephemeral keys for admins | admin kits, long-lived admin credentials | PR #4846 open | admin cert lifetime |
| 2 | Ephemeral keys for server/client | key distribution, restart-per-renewal | branch + renewal spike | — (30-day certs decided, D1) |
| 3 | Ephemeral keys for jobs | launch-time auth tokens (the site's own bearer, on disk / ps), shared site key across jobs | **prototype** (job CA, SP-issued at deploy) | job CA: keep vs chain under external issuer |
| 4 | KMS-backed provisioning (trust-only) | root + participant keys in provisioning state | not started, seam exists (BYO root) | not-yet-enrolled kit behavior |
| 5 | Key rotation for server/client | unbounded key lifetime | procedure only, no new code | D3: only if a deployment can never restart its server — build on demand (D2 decided: rotate ≤ 1 y) |
| 6 | Study entitlements in admin certs | per-study admin-list maintenance for SSO users | designed (`sso-study.md`), not started | extension OID + encoding; malformed-claim scope |

**Base posture:** external CA (step-ca) + KMS root, short-lived certs renewed
live, keys rotated at restarts. FLARE-as-CA (the baseline) remains for POCs.

**Deliberately not built:** live key-swap, revocation infrastructure,
CA/issuance inside FLARE.
