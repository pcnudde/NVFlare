------------------------- MODULE JobOutcomeProtocol -------------------------
EXTENDS Naturals

(***************************************************************************
Interface model for one client process and one server completion owner.

The client only promises that it attempted a terminal report before releasing
its resources.  Delivery is not guaranteed: the server must either accept an
authenticated participant report or resolve the obligation through its
missing-client/timeout fallback.  Report duplication is harmless, a client
failure dominates candidate server success, and a server abort eventually
reaches the client under the stated fairness assumption.

The Boolean constants are deliberate mutations.  Safe configurations set all
of them TRUE; each unsafe configuration disables one protocol rule so TLC
must produce a counterexample or liveness failure.
***************************************************************************)

CONSTANTS EnforceIdentity, AcceptValidReport, RequireResolutionBeforePublish,
          RequireReportBeforeRelease, FailureDominates,
          IdempotentSettlement, ReliableAbort

ASSUME /\ EnforceIdentity \in BOOLEAN
       /\ AcceptValidReport \in BOOLEAN
       /\ RequireResolutionBeforePublish \in BOOLEAN
       /\ RequireReportBeforeRelease \in BOOLEAN
       /\ FailureDominates \in BOOLEAN
       /\ IdempotentSettlement \in BOOLEAN
       /\ ReliableAbort \in BOOLEAN

ClientPhases == {"owned", "exited", "reporting", "settled", "released"}
Outcomes == {"none", "no_override", "failure"}
ReportResults == {"none", "lost", "accepted", "rejected"}
ServerPhases == {"running", "waiting", "publishing", "done"}
Candidates == {"none", "completed", "failed"}
Obligations == {"pending", "report", "fallback"}
Senders == {"none", "participant", "attacker"}
PublishedStatuses == {"none", "completed", "failed"}
AbortStates == {"none", "pending", "delivered"}

VARIABLES clientPhase, clientOutcome, reportAttempted, participantCopies,
          reportDuplicated, reportResult, attackerMessage, serverPhase, candidate, obligation,
          acceptedSender, acceptedOutcome, resolutionCount, published, abort

vars == <<clientPhase, clientOutcome, reportAttempted, participantCopies,
          reportDuplicated, reportResult, attackerMessage, serverPhase, candidate, obligation,
          acceptedSender, acceptedOutcome, resolutionCount, published, abort>>

Init ==
    /\ clientPhase = "owned"
    /\ clientOutcome = "none"
    /\ reportAttempted = FALSE
    /\ participantCopies = 0
    /\ reportDuplicated = FALSE
    /\ reportResult = "none"
    /\ attackerMessage = "none"
    /\ serverPhase = "running"
    /\ candidate = "none"
    /\ obligation = "pending"
    /\ acceptedSender = "none"
    /\ acceptedOutcome = "none"
    /\ resolutionCount = 0
    /\ published = "none"
    /\ abort = "none"

ClientExited(outcome) ==
    /\ clientPhase = "owned"
    /\ outcome \in Outcomes \ {"none"}
    /\ clientPhase' = "exited"
    /\ clientOutcome' = outcome
    /\ UNCHANGED <<reportAttempted, participantCopies, reportDuplicated, reportResult,
                    attackerMessage, serverPhase, candidate, obligation,
                    acceptedSender, acceptedOutcome, resolutionCount,
                    published, abort>>

StartParticipantReport ==
    /\ clientPhase = "exited"
    /\ clientPhase' = "reporting"
    /\ reportAttempted' = TRUE
    /\ participantCopies' = 1
    /\ reportResult' = "none"
    /\ UNCHANGED <<clientOutcome, reportDuplicated, attackerMessage, serverPhase, candidate,
                    obligation, acceptedSender, acceptedOutcome,
                    resolutionCount, published, abort>>

DuplicateParticipantReport ==
    /\ clientPhase = "reporting"
    /\ participantCopies = 1
    /\ ~reportDuplicated
    /\ participantCopies' = 2
    /\ reportDuplicated' = TRUE
    /\ UNCHANGED <<clientPhase, clientOutcome, reportAttempted, reportResult,
                    attackerMessage, serverPhase, candidate, obligation,
                    acceptedSender, acceptedOutcome, resolutionCount,
                    published, abort>>

LoseParticipantReport ==
    /\ participantCopies > 0
    /\ participantCopies' = participantCopies - 1
    /\ reportResult' = IF participantCopies = 1 /\ reportResult = "none"
                           THEN "lost"
                           ELSE reportResult
    /\ UNCHANGED <<clientPhase, clientOutcome, reportAttempted, reportDuplicated,
                    attackerMessage, serverPhase, candidate, obligation,
                    acceptedSender, acceptedOutcome, resolutionCount,
                    published, abort>>

DeliverParticipantReport ==
    /\ participantCopies > 0
    /\ participantCopies' = participantCopies - 1
    /\ IF obligation = "pending"
          THEN IF AcceptValidReport
                  THEN /\ obligation' = "report"
                       /\ acceptedSender' = "participant"
                       /\ acceptedOutcome' = clientOutcome
                       /\ resolutionCount' = 1
                       /\ reportResult' = "accepted"
                       /\ serverPhase' = IF serverPhase = "waiting"
                                             THEN "publishing"
                                             ELSE serverPhase
                  ELSE /\ reportResult' = "rejected"
                       /\ UNCHANGED <<obligation, acceptedSender, acceptedOutcome,
                                        resolutionCount, serverPhase>>
          ELSE /\ UNCHANGED <<obligation, acceptedSender, acceptedOutcome>>
               /\ resolutionCount' = IF IdempotentSettlement
                                         THEN resolutionCount
                                         ELSE resolutionCount + 1
               /\ reportResult' = IF reportResult = "accepted"
                                      THEN "accepted"
                                      ELSE "rejected"
               /\ UNCHANGED serverPhase
    /\ UNCHANGED <<clientPhase, clientOutcome, reportAttempted, reportDuplicated,
                    attackerMessage, candidate, published, abort>>

FinishParticipantReport ==
    /\ clientPhase = "reporting"
    /\ reportResult # "none"
    /\ clientPhase' = "settled"
    /\ UNCHANGED <<clientOutcome, reportAttempted, participantCopies, reportDuplicated,
                    reportResult, attackerMessage, serverPhase, candidate,
                    obligation, acceptedSender, acceptedOutcome,
                    resolutionCount, published, abort>>

ReleaseClient ==
    /\ \/ clientPhase = "settled"
       \/ ~RequireReportBeforeRelease /\ clientPhase = "exited"
    /\ clientPhase' = "released"
    /\ UNCHANGED <<clientOutcome, reportAttempted, participantCopies, reportDuplicated,
                    reportResult, attackerMessage, serverPhase, candidate,
                    obligation, acceptedSender, acceptedOutcome,
                    resolutionCount, published, abort>>

SendAttackerReport(outcome) ==
    /\ serverPhase # "done"
    /\ attackerMessage = "none"
    /\ outcome \in Outcomes \ {"none"}
    /\ attackerMessage' = outcome
    /\ UNCHANGED <<clientPhase, clientOutcome, reportAttempted,
                    participantCopies, reportDuplicated, reportResult, serverPhase, candidate,
                    obligation, acceptedSender, acceptedOutcome,
                    resolutionCount, published, abort>>

DeliverAttackerReport ==
    /\ attackerMessage # "none"
    /\ attackerMessage' = "none"
    /\ IF ~EnforceIdentity /\ obligation = "pending"
          THEN /\ obligation' = "report"
               /\ acceptedSender' = "attacker"
               /\ acceptedOutcome' = attackerMessage
               /\ resolutionCount' = 1
               /\ serverPhase' = IF serverPhase = "waiting"
                                     THEN "publishing"
                                     ELSE serverPhase
          ELSE /\ UNCHANGED <<obligation, acceptedSender, acceptedOutcome,
                               resolutionCount, serverPhase>>
    /\ UNCHANGED <<clientPhase, clientOutcome, reportAttempted,
                    participantCopies, reportDuplicated, reportResult, candidate, published,
                    abort>>

ServerExited(result) ==
    /\ serverPhase = "running"
    /\ result \in Candidates \ {"none"}
    /\ candidate' = result
    /\ serverPhase' = IF obligation = "pending" THEN "waiting" ELSE "publishing"
    /\ UNCHANGED <<clientPhase, clientOutcome, reportAttempted,
                    participantCopies, reportDuplicated, reportResult, attackerMessage,
                    obligation, acceptedSender, acceptedOutcome,
                    resolutionCount, published, abort>>

ResolveByFallback ==
    /\ serverPhase = "waiting"
    /\ obligation = "pending"
    /\ serverPhase' = "publishing"
    /\ obligation' = "fallback"
    /\ resolutionCount' = 1
    /\ UNCHANGED <<clientPhase, clientOutcome, reportAttempted,
                    participantCopies, reportDuplicated, reportResult, attackerMessage,
                    candidate, acceptedSender, acceptedOutcome, published,
                    abort>>

PublishStatus ==
    /\ candidate # "none"
    /\ \/ serverPhase = "publishing"
       \/ ~RequireResolutionBeforePublish /\ serverPhase = "waiting"
    /\ serverPhase' = "done"
    /\ published' =
          IF candidate = "failed"
             \/ obligation = "fallback"
             \/ (acceptedOutcome = "failure" /\ FailureDominates)
          THEN "failed"
          ELSE "completed"
    /\ UNCHANGED <<clientPhase, clientOutcome, reportAttempted,
                    participantCopies, reportDuplicated, reportResult, attackerMessage,
                    candidate, obligation, acceptedSender, acceptedOutcome,
                    resolutionCount, abort>>

RequestAbort ==
    /\ abort = "none"
    /\ clientPhase = "owned"
    /\ abort' = "pending"
    /\ UNCHANGED <<clientPhase, clientOutcome, reportAttempted,
                    participantCopies, reportDuplicated, reportResult, attackerMessage,
                    serverPhase, candidate, obligation, acceptedSender,
                    acceptedOutcome, resolutionCount, published>>

DeliverAbort ==
    /\ ReliableAbort
    /\ abort = "pending"
    /\ abort' = "delivered"
    /\ UNCHANGED <<clientPhase, clientOutcome, reportAttempted,
                    participantCopies, reportDuplicated, reportResult, attackerMessage,
                    serverPhase, candidate, obligation, acceptedSender,
                    acceptedOutcome, resolutionCount, published>>

ExitAfterAbort ==
    /\ abort = "delivered"
    /\ clientPhase = "owned"
    /\ clientPhase' = "exited"
    /\ clientOutcome' = "failure"
    /\ UNCHANGED <<reportAttempted, participantCopies, reportDuplicated, reportResult,
                    attackerMessage, serverPhase, candidate, obligation,
                    acceptedSender, acceptedOutcome, resolutionCount,
                    published, abort>>

ParticipantNetworkStep == DeliverParticipantReport \/ LoseParticipantReport

Next ==
    \/ \E outcome \in Outcomes \ {"none"} : ClientExited(outcome)
    \/ StartParticipantReport
    \/ DuplicateParticipantReport
    \/ ParticipantNetworkStep
    \/ FinishParticipantReport
    \/ ReleaseClient
    \/ \E outcome \in Outcomes \ {"none"} : SendAttackerReport(outcome)
    \/ DeliverAttackerReport
    \/ \E result \in Candidates \ {"none"} : ServerExited(result)
    \/ ResolveByFallback
    \/ PublishStatus
    \/ RequestAbort
    \/ DeliverAbort
    \/ ExitAfterAbort

Fairness ==
    /\ WF_vars(StartParticipantReport)
    /\ WF_vars(ParticipantNetworkStep)
    /\ WF_vars(FinishParticipantReport)
    /\ WF_vars(ReleaseClient)
    /\ WF_vars(ResolveByFallback)
    /\ WF_vars(PublishStatus)
    /\ WF_vars(DeliverAbort)
    /\ WF_vars(ExitAfterAbort)

Spec == Init /\ [][Next]_vars /\ Fairness

TypeOK ==
    /\ clientPhase \in ClientPhases
    /\ clientOutcome \in Outcomes
    /\ reportAttempted \in BOOLEAN
    /\ participantCopies \in 0..2
    /\ reportDuplicated \in BOOLEAN
    /\ reportResult \in ReportResults
    /\ attackerMessage \in Outcomes
    /\ serverPhase \in ServerPhases
    /\ candidate \in Candidates
    /\ obligation \in Obligations
    /\ acceptedSender \in Senders
    /\ acceptedOutcome \in Outcomes
    /\ resolutionCount \in 0..2
    /\ published \in PublishedStatuses
    /\ abort \in AbortStates

ResolutionHasEvidence ==
    /\ (obligation = "pending")
          => acceptedSender = "none" /\ acceptedOutcome = "none" /\ resolutionCount = 0
    /\ (obligation = "report")
          => acceptedSender # "none" /\ acceptedOutcome # "none" /\ resolutionCount >= 1
    /\ (obligation = "fallback")
          => acceptedSender = "none" /\ acceptedOutcome = "none" /\ resolutionCount = 1

AcceptedReportIsAuthenticated ==
    obligation = "report" => acceptedSender = "participant"

DeliveredValidReportIsAccepted ==
    reportResult = "rejected" => obligation # "pending"

CompletedPublicationIsSafe ==
    published = "completed"
        => /\ candidate = "completed"
           /\ obligation = "report"
           /\ acceptedSender = "participant"
           /\ acceptedOutcome = "no_override"

ClientReleaseFollowsReportAttempt ==
    clientPhase = "released" => reportAttempted

FailureDominatesPublication ==
    acceptedOutcome = "failure" => published # "completed"

AtMostOneSettlement == resolutionCount <= 1

ServerProgress == (serverPhase = "waiting") ~> (serverPhase = "done")

ClientCompletionProgress ==
    (clientPhase = "exited") ~> (clientPhase = "released")

AbortProgress ==
    (abort = "pending") ~> (clientPhase # "owned")

=============================================================================
