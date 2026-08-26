---------------------------- MODULE JobCompletion ----------------------------
EXTENDS Naturals

(***************************************************************************
This is the bounded counterpart of nvflare/private/fed/server/job_completion.py.
JobCompletionDriver supplies clock-expiry and I/O-result events through effects
implemented by JobRunner.  This model owns their ordering and the rules that
downgrade candidate success.

Four Boolean constants deliberately recreate historical failure shapes:
publishing success before archive, weakening a terminal failure, re-archiving
after a committed archive, and retrying an effect forever.  The last mutation
toggles retryPulse so an infinite retry execution is visible to TLC rather than
being treated as stuttering.
***************************************************************************)

CONSTANTS Clients, SafePublication, SafeStatusPrecedence,
          RetryCleanupByRearchive, RetryForever, RetryLimit

ASSUME /\ Clients # {}
       /\ SafePublication \in BOOLEAN
       /\ SafeStatusPrecedence \in BOOLEAN
       /\ RetryCleanupByRearchive \in BOOLEAN
       /\ RetryForever \in BOOLEAN
       /\ RetryLimit \in Nat

Phases == {
    "waiting_for_server", "waiting_for_clients", "archiving", "cleaning",
    "publishing", "done"
}
Statuses == {"none", "completed", "aborted", "failed", "abnormal"}
TerminalStatuses == Statuses \ {"none"}
NonSuccessStatuses == TerminalStatuses \ {"completed"}
ClientOutcomes == {"no_override", "execution_failure", "abnormal", "aborted", "unsafe"}

VARIABLES phase, pendingClients, status, archiveCommitted,
          retriesLeft, retryPulse

vars == <<phase, pendingClients, status, archiveCommitted,
          retriesLeft, retryPulse>>

Init ==
    /\ phase = "waiting_for_server"
    /\ pendingClients = Clients
    /\ status = "none"
    /\ archiveCommitted = FALSE
    /\ retriesLeft = RetryLimit
    /\ retryPulse = FALSE

OutcomeStatus(outcome) ==
    CASE outcome = "execution_failure" -> "failed"
      [] outcome = "abnormal" -> "abnormal"
      [] outcome \in {"aborted", "unsafe"} -> "aborted"

StatusRank(s) ==
    CASE s = "none" -> 0
      [] s = "completed" -> 1
      [] s = "aborted" -> 2
      [] s = "failed" -> 3
      [] s = "abnormal" -> 4

MergeStatus(current, candidate) ==
    IF ~SafeStatusPrecedence \/ StatusRank(candidate) > StatusRank(current)
       THEN candidate
       ELSE current

ParticipantsSelected(active) ==
    /\ phase = "waiting_for_server"
    /\ active \subseteq Clients
    /\ pendingClients' = pendingClients \cap active
    /\ UNCHANGED <<phase, status, archiveCommitted, retriesLeft, retryPulse>>

RecordClientOutcome(c, outcome) ==
    /\ phase \in {"waiting_for_server", "waiting_for_clients"}
    /\ c \in pendingClients
    /\ outcome \in ClientOutcomes
    /\ IF outcome = "no_override"
          THEN /\ pendingClients' = pendingClients \ {c}
               /\ phase' = IF phase = "waiting_for_clients" /\ pendingClients' = {}
                               THEN "archiving"
                               ELSE phase
               /\ UNCHANGED status
          ELSE /\ pendingClients' = {}
               /\ phase' = IF phase = "waiting_for_clients" THEN "archiving" ELSE phase
               /\ status' = MergeStatus(status, OutcomeStatus(outcome))
    /\ UNCHANGED <<archiveCommitted, retriesLeft, retryPulse>>

ServerExited(s) ==
    /\ phase = "waiting_for_server"
    /\ s \in TerminalStatuses
    /\ status' = MergeStatus(status, s)
    /\ phase' = IF pendingClients = {} THEN "archiving" ELSE "waiting_for_clients"
    /\ UNCHANGED <<pendingClients, archiveCommitted, retriesLeft, retryPulse>>

TerminalOverride(s) ==
    /\ phase \in {"waiting_for_server", "waiting_for_clients"}
    /\ s \in NonSuccessStatuses
    /\ phase' = IF phase = "waiting_for_clients" THEN "archiving" ELSE phase
    /\ pendingClients' = {}
    /\ status' = MergeStatus(status, s)
    /\ UNCHANGED <<archiveCommitted, retriesLeft, retryPulse>>

ClientWaitExpired ==
    /\ phase = "waiting_for_clients"
    /\ phase' = "archiving"
    /\ pendingClients' = {}
    /\ status' = IF status = "completed" THEN "failed" ELSE status
    /\ UNCHANGED <<archiveCommitted, retriesLeft, retryPulse>>

ArchiveRetry ==
    /\ phase = "archiving"
    /\ IF RetryForever
          THEN retriesLeft' = retriesLeft
          ELSE /\ retriesLeft > 0
               /\ retriesLeft' = retriesLeft - 1
    /\ retryPulse' = ~retryPulse
    /\ UNCHANGED <<phase, pendingClients, status, archiveCommitted>>

ArchiveCommitted ==
    /\ phase = "archiving"
    /\ phase' = "cleaning"
    /\ archiveCommitted' = TRUE
    /\ retriesLeft' = RetryLimit
    /\ UNCHANGED <<pendingClients, status, retryPulse>>

ArchiveAbandoned ==
    /\ phase = "archiving"
    /\ phase' = "publishing"
    /\ status' = IF status = "completed" /\ SafePublication THEN "failed" ELSE status
    /\ UNCHANGED <<pendingClients, archiveCommitted, retriesLeft, retryPulse>>

CleanupRetry ==
    /\ phase = "cleaning"
    /\ IF RetryForever
          THEN retriesLeft' = retriesLeft
          ELSE /\ retriesLeft > 0
               /\ retriesLeft' = retriesLeft - 1
    /\ retryPulse' = ~retryPulse
    /\ UNCHANGED <<phase, pendingClients, status, archiveCommitted>>

CleanupSettled ==
    /\ phase = "cleaning"
    /\ phase' = "publishing"
    /\ UNCHANGED <<pendingClients, status, archiveCommitted, retriesLeft, retryPulse>>

CleanupRetryRearchives ==
    /\ RetryCleanupByRearchive
    /\ phase = "cleaning"
    /\ phase' = "archiving"
    /\ UNCHANGED <<pendingClients, status, archiveCommitted,
                    retriesLeft, retryPulse>>

StatusPublished ==
    /\ phase = "publishing"
    /\ IF SafePublication THEN status # "completed" \/ archiveCommitted ELSE TRUE
    /\ phase' = "done"
    /\ UNCHANGED <<pendingClients, status, archiveCommitted, retriesLeft, retryPulse>>

ArchiveStep == ArchiveRetry \/ ArchiveCommitted \/ ArchiveAbandoned

CleanupStep == CleanupRetry \/ CleanupSettled \/ CleanupRetryRearchives

Next ==
    \/ \E active \in SUBSET Clients : ParticipantsSelected(active)
    \/ \E c \in Clients, outcome \in ClientOutcomes : RecordClientOutcome(c, outcome)
    \/ \E s \in TerminalStatuses : ServerExited(s)
    \/ \E s \in NonSuccessStatuses : TerminalOverride(s)
    \/ ClientWaitExpired
    \/ ArchiveRetry
    \/ ArchiveCommitted
    \/ ArchiveAbandoned
    \/ CleanupRetry
    \/ CleanupSettled
    \/ CleanupRetryRearchives
    \/ StatusPublished

Fairness ==
    /\ WF_vars(ClientWaitExpired)
    /\ WF_vars(ArchiveStep)
    /\ WF_vars(CleanupStep)
    /\ WF_vars(StatusPublished)

Spec == Init /\ [][Next]_vars /\ Fairness

TypeOK ==
    /\ phase \in Phases
    /\ pendingClients \subseteq Clients
    /\ status \in Statuses
    /\ archiveCommitted \in BOOLEAN
    /\ retriesLeft \in 0..RetryLimit
    /\ retryPulse \in BOOLEAN

ServerObserved == phase # "waiting_for_server" => status # "none"

ClientsSettledBeforeArchive ==
    phase \in {"archiving", "cleaning", "publishing", "done"}
        => pendingClients = {}

CompletedPublicationHasArchive ==
    phase = "done" /\ status = "completed" => archiveCommitted

CommittedArchiveIsNotRewritten ==
    archiveCommitted => phase # "archiving"

TerminalProgress ==
    (phase # "waiting_for_server") ~> (phase = "done")

TerminalStatusMonotonic ==
    /\ [] (status = "abnormal" => [] (status = "abnormal"))
    /\ [] (status = "failed" => [] (status \in {"failed", "abnormal"}))
    /\ [] (status = "aborted" => [] (status \in {"aborted", "failed", "abnormal"}))

=============================================================================
