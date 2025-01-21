(define (problem SMALL-GRAPH)
    (:domain GRAPH)
    (:objects A B C D E)
    (:init 
        (agent-at A)

        (arc A B)
        (arc B C)
        (arc C D)
        (arc D E)
    )
    (:goal (agent-at E))
)