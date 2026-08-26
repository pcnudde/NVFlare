----------------------------- MODULE ClientProcess ----------------------------
EXTENDS Naturals

(***************************************************************************
Bounded model of the client-owned job process in client_process.py.

The safe core maps exactly to the Python transition graph.  Ghost variables
remember accepted stop intent and cleanup facts so deliberately unsafe rules
cannot erase their history.  Optional termination retry actions model the
side-effect driver and are disabled during exact core-graph comparison.
***************************************************************************)

CONSTANTS SafeAttachStop, SafeStopPrecedence, StoppedKeepsOwnership,
          SafeCompletionOrder, ModelTermination, RetryForever, RetryLimit

ASSUME /\ SafeAttachStop \in BOOLEAN
       /\ SafeStopPrecedence \in BOOLEAN
       /\ StoppedKeepsOwnership \in BOOLEAN
       /\ SafeCompletionOrder \in BOOLEAN
       /\ ModelTermination \in BOOLEAN
       /\ RetryForever \in BOOLEAN
       /\ RetryLimit \in Nat

Phases == {
    "launching", "running", "runner_stopped", "exited", "outcome_settled",
    "resources_released", "unregistered", "done", "launch_failed"
}
OwnedPhases == {"launching", "running", "runner_stopped"}
StopIntents == {"none", "heartbeat_cleanup", "user_abort"}
RealStopIntents == StopIntents \ {"none"}

StopRank(i) ==
    CASE i = "none" -> 0
      [] i = "heartbeat_cleanup" -> 1
      [] i = "user_abort" -> 2

StrongerStop(a, b) == IF StopRank(a) >= StopRank(b) THEN a ELSE b

VARIABLES phase, handleAttached, stopIntent,
          acceptedStop, exitObserved, resourcesReleased, registered,
          terminationAccepted, retriesLeft, retryPulse

vars == <<phase, handleAttached, stopIntent,
          acceptedStop, exitObserved, resourcesReleased, registered,
          terminationAccepted, retriesLeft, retryPulse>>

Init ==
    /\ phase = "launching"
    /\ handleAttached = FALSE
    /\ stopIntent = "none"
    /\ acceptedStop = "none"
    /\ exitObserved = FALSE
    /\ resourcesReleased = FALSE
    /\ registered = TRUE
    /\ terminationAccepted = FALSE
    /\ retriesLeft = RetryLimit
    /\ retryPulse = FALSE

AttachHandle ==
    /\ phase \in OwnedPhases
    /\ ~handleAttached
    /\ handleAttached' = TRUE
    /\ stopIntent' = IF SafeAttachStop THEN stopIntent ELSE "none"
    /\ UNCHANGED <<phase, acceptedStop, exitObserved, resourcesReleased,
                    registered, terminationAccepted, retriesLeft, retryPulse>>

WorkerStarted ==
    /\ phase = "launching"
    /\ phase' = "running"
    /\ UNCHANGED <<handleAttached, stopIntent, acceptedStop, exitObserved,
                    resourcesReleased, registered, terminationAccepted,
                    retriesLeft, retryPulse>>

WorkerStopped ==
    /\ phase \in {"launching", "running"}
    /\ IF StoppedKeepsOwnership
          THEN /\ phase' = "runner_stopped"
               /\ registered' = registered
          ELSE /\ phase' = "unregistered"
               /\ registered' = FALSE
    /\ UNCHANGED <<handleAttached, stopIntent, acceptedStop, exitObserved,
                    resourcesReleased, terminationAccepted, retriesLeft,
                    retryPulse>>

RequestStop(i) ==
    /\ phase \in OwnedPhases
    /\ i \in RealStopIntents
    /\ stopIntent' = IF SafeStopPrecedence THEN StrongerStop(stopIntent, i) ELSE i
    /\ acceptedStop' = StrongerStop(acceptedStop, i)
    /\ UNCHANGED <<phase, handleAttached, exitObserved, resourcesReleased,
                    registered, terminationAccepted, retriesLeft, retryPulse>>

ProcessExited ==
    /\ phase \in OwnedPhases
    /\ handleAttached
    /\ phase' = "exited"
    /\ exitObserved' = TRUE
    /\ stopIntent' = "none"
    /\ UNCHANGED <<handleAttached, acceptedStop, resourcesReleased,
                    registered, terminationAccepted, retriesLeft, retryPulse>>

OutcomeSettled ==
    /\ phase = "exited"
    /\ phase' = "outcome_settled"
    /\ UNCHANGED <<handleAttached, stopIntent, acceptedStop, exitObserved,
                    resourcesReleased, registered, terminationAccepted,
                    retriesLeft, retryPulse>>

ResourcesReleased ==
    /\ phase = "outcome_settled"
    /\ phase' = "resources_released"
    /\ resourcesReleased' = TRUE
    /\ UNCHANGED <<handleAttached, stopIntent, acceptedStop, exitObserved,
                    registered, terminationAccepted, retriesLeft, retryPulse>>

Unregistered ==
    /\ phase = "resources_released"
    /\ phase' = "unregistered"
    /\ registered' = FALSE
    /\ UNCHANGED <<handleAttached, stopIntent, acceptedStop, exitObserved,
                    resourcesReleased, terminationAccepted, retriesLeft,
                    retryPulse>>

CompletionPublished ==
    /\ IF SafeCompletionOrder
          THEN phase = "unregistered"
          ELSE phase \in {"outcome_settled", "unregistered"}
    /\ phase' = "done"
    /\ UNCHANGED <<handleAttached, stopIntent, acceptedStop, exitObserved,
                    resourcesReleased, registered, terminationAccepted,
                    retriesLeft, retryPulse>>

LaunchFailed ==
    /\ phase \in OwnedPhases
    /\ ~handleAttached
    /\ phase' = "launch_failed"
    /\ stopIntent' = "none"
    /\ acceptedStop' = "none"
    /\ registered' = FALSE
    /\ UNCHANGED <<handleAttached, exitObserved,
                    resourcesReleased, terminationAccepted, retriesLeft,
                    retryPulse>>

TerminationRetry ==
    /\ ModelTermination
    /\ phase \in OwnedPhases
    /\ handleAttached
    /\ acceptedStop # "none"
    /\ ~terminationAccepted
    /\ IF RetryForever
          THEN retriesLeft' = retriesLeft
          ELSE /\ retriesLeft > 0
               /\ retriesLeft' = retriesLeft - 1
    /\ retryPulse' = ~retryPulse
    /\ UNCHANGED <<phase, handleAttached, stopIntent, acceptedStop,
                    exitObserved, resourcesReleased, registered,
                    terminationAccepted>>

TerminationAccepted ==
    /\ ModelTermination
    /\ phase \in OwnedPhases
    /\ handleAttached
    /\ acceptedStop # "none"
    /\ ~terminationAccepted
    /\ terminationAccepted' = TRUE
    /\ UNCHANGED <<phase, handleAttached, stopIntent, acceptedStop,
                    exitObserved, resourcesReleased, registered, retriesLeft,
                    retryPulse>>

TerminatedExit ==
    /\ ModelTermination
    /\ phase \in OwnedPhases
    /\ handleAttached
    /\ terminationAccepted
    /\ phase' = "exited"
    /\ exitObserved' = TRUE
    /\ stopIntent' = "none"
    /\ UNCHANGED <<handleAttached, acceptedStop, resourcesReleased,
                    registered, terminationAccepted, retriesLeft, retryPulse>>

TerminationStep == TerminationRetry \/ TerminationAccepted

Next ==
    \/ AttachHandle
    \/ WorkerStarted
    \/ WorkerStopped
    \/ \E i \in RealStopIntents : RequestStop(i)
    \/ ProcessExited
    \/ OutcomeSettled
    \/ ResourcesReleased
    \/ Unregistered
    \/ CompletionPublished
    \/ LaunchFailed
    \/ TerminationRetry
    \/ TerminationAccepted
    \/ TerminatedExit

Fairness ==
    /\ WF_vars(TerminationStep)
    /\ WF_vars(TerminatedExit)
    /\ WF_vars(OutcomeSettled)
    /\ WF_vars(ResourcesReleased)
    /\ WF_vars(Unregistered)
    /\ WF_vars(CompletionPublished)

Spec == Init /\ [][Next]_vars /\ Fairness

TypeOK ==
    /\ phase \in Phases
    /\ handleAttached \in BOOLEAN
    /\ stopIntent \in StopIntents
    /\ acceptedStop \in StopIntents
    /\ exitObserved \in BOOLEAN
    /\ resourcesReleased \in BOOLEAN
    /\ registered \in BOOLEAN
    /\ terminationAccepted \in BOOLEAN
    /\ retriesLeft \in 0..RetryLimit
    /\ retryPulse \in BOOLEAN

AcceptedStopPreserved ==
    phase \in OwnedPhases => StopRank(stopIntent) = StopRank(acceptedStop)

StoppedStillOwned ==
    phase = "runner_stopped" => registered /\ ~exitObserved

RemovalAfterExit ==
    ~registered => exitObserved \/ phase = "launch_failed"

ResourcesAfterExit == resourcesReleased => exitObserved

PostExitHasHandle == exitObserved => handleAttached

CompletionIsClean ==
    phase = "done" => exitObserved /\ resourcesReleased /\ ~registered

RegistrationMatchesPhase ==
    registered = ~(phase \in {"unregistered", "done", "launch_failed"})

ExitEventuallyDone == exitObserved ~> (phase = "done")

StopEventuallyDone ==
    (acceptedStop # "none" /\ handleAttached) ~> (phase = "done")

=============================================================================
