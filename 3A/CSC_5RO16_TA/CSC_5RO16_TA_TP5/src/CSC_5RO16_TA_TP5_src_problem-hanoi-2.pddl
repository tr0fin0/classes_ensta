(define (problem HANOI-2)
	(:domain HANOI)
	(:objects T1 T2 T3 D1 D2)
	(:init

		(tower T1)	(tower T2)	(tower T3)
		(disk D1)	(disk D2)
		(smaller D1 T1)	(smaller D2 T1)
		(smaller D1 T2)	(smaller D2 T2)
		(smaller D1 T3)	(smaller D2 T3)
		(smaller D1 D2)
		
		(on D1 D2)	(on D2 T1)
		(clear D1)	(clear T2)	(clear T3)

		(tower_empty)
	)
	(:goal 	(and (on D1 D2) (on D2 T3))
	)
)