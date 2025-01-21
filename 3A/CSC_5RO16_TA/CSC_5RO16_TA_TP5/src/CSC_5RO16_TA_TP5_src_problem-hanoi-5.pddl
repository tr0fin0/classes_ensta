(define (problem HANOI-5)
	(:domain HANOI)
	(:objects T1 T2 T3 D1 D2 D3 D4 D5)
	(:init

		(tower T1)	(tower T2)	(tower T3)
		(disk D1)	(disk D2)	(disk D3)	(disk D4)	(disk D5)
		(smaller D1 T1)	(smaller D2 T1)	(smaller D3 T1)	(smaller D4 T1)	(smaller D5 T1)
		(smaller D1 T2)	(smaller D2 T2)	(smaller D3 T2)	(smaller D4 T2)	(smaller D5 T2)
		(smaller D1 T3)	(smaller D2 T3)	(smaller D3 T3)	(smaller D4 T3)	(smaller D5 T3)
		(smaller D1 D2)	(smaller D1 D3)	(smaller D1 D4)	(smaller D1 D5)
		(smaller D2 D3)	(smaller D2 D4)	(smaller D2 D5)
		(smaller D3 D4)	(smaller D3 D5)
		(smaller D4 D5)
		
		(on D1 D2)	(on D2 D3)	(on D3 D4)	(on D4 D5)	(on D5 T1)
		(clear D1)	(clear T2)	(clear T3)

		(tower_empty)
	)
	(:goal 	(and (on D1 D2) (on D2 D3) (on D3 D4) (on D4 D5) (on D5 T3))
	)
)