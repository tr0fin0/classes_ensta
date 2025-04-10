(define (problem MONKEY-BANANA)
    (:domain MONKEY)
    (:objects A B C bananes)
    (:init 
        (singe A)
        (caisse B)
        (bananes C)
        (singe-bas)
        (vide)
    )
    (:goal 
        (and 
            (singe C)
            (caisse C)
            (attrape bananes)
        )
    )
)