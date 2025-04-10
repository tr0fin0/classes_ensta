    (define (problem BLOCKS-MULTI-TOWERS)
        (:domain BLOCKS)
        (:objects A B C D E F G H I J)
        (:init 
            ;; Tour 1
            (on C G) (on G E) (on E I) (on I J) (on J A) (on A B) (ontable B)
            ;; Tour 2
            (on F D) (on D H) (ontable H)
            ;; Propriétés initiales
            (clear C) (clear F) (handempty)
        )
        (:goal 
            (and 
                (on C B) (on B D) (on D F)
                (on F I) (on I A) (on A E)
                (on E H) (on H G) (on G J)
            )
        )
    )