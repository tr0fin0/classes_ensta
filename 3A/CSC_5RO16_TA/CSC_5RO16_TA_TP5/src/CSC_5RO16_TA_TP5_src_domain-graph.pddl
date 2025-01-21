(define (domain GRAPH)
    (:requirements :strips)
    (:predicates 
        (agent-at ?node)
        (arc ?from ?to)
    )
    (:action move
        :parameters (?from ?to)
        :precondition (and (agent-at ?from) (arc ?from ?to))
        :effect (and 
            (not (agent-at ?from))
            (agent-at ?to)
        )
    )
)