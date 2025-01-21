(define (domain MONKEY)
    (:requirements :strips)
    (:predicates
        (vide)
        (singe ?from)
        (singe-bas)
        (singe-caisse ?from)
        (caisse ?from)
        (bananes ?from)
        (attrape ?objet)
    )
    (:action aller
        :parameters (?from ?to)
        :precondition (and
            (singe ?from)
            (singe-bas)
        )
        :effect (and 
            (not (singe ?from))
            (singe ?to)
        )
    )
    (:action pousser
        :parameters (?objet ?from ?to)
        :precondition (and 
            (singe ?from) 
            (caisse ?from)
            (singe-bas)
        )
        :effect (and 
            (not (caisse ?from))
            (caisse ?to)
            (not (singe ?from))
            (singe ?to)
        )
    )
    (:action monter
        :parameters (?lieu)
        :precondition (and 
            (singe ?lieu) 
            (caisse ?lieu) 
            (singe-bas)
        )
        :effect (and
            (not (singe-bas))
            (singe-caisse ?lieu)
        )
    )
    (:action descendre
        :parameters (?objet ?lieu)
        :precondition (and 
            (singe ?lieu) 
            (caisse ?lieu) 
            (singe-caisse ?lieu)
        )
        :effect (and
            (singe-bas)
            (not (singe-caisse ?lieu))
        )
    )
    (:action attraper
        :parameters (?objet ?lieu)
        :precondition (and 
            (singe ?lieu) 
            (bananes ?lieu) 
            (singe-caisse ?lieu) 
            (vide)
        )
        :effect (and 
            (not (vide))
            (attrape ?objet)
        )
    )
    (:action lacher
        :parameters (?objet ?lieu)
        :precondition (and 
            (attrape ?objet) 
            (singe ?lieu)
        )
        :effect (and 
            (not (attrape ?objet))
            (vide)
        )
    )
)