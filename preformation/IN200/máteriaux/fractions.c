#include <stdio.h>
#include <string.h>
#include <stdbool.h>



/** Sign of a fraction. */
enum sign_t { S_pos, S_neg } ;


/** Representation of a fraction. no use of signed integers to avoid
    inconsistencies with the field 'sign'. */
struct fraction_t {
  enum sign_t sign ;           /* Sign. */
  unsigned long num ;          /* Numerator. */
  unsigned long denom ;        /* Denominator. */
} ;


/** Print a fraction. No sign is printed if the fraction is positive.
    No denominator is printed if the fraction represents an integer. */
void fr_print (struct fraction_t fract)
{
  if (fract.num == 0) printf ("0") ;
  else {
    /* Handle the sign. */
    if (fract.sign == S_neg) printf ("-") ;
    /* Print the absolute value of the fraction. */
    if (fract.denom == 1) printf ("%ld", fract.num) ;
    else printf ("%ld/%ld", fract.num, fract.denom) ;
  }
}


/** Add two fractions 'fract1' and 'fract2' and set the result in 'fract3' by
    reference passing style. Returns 'true' if the operation is a success,
    'false' otherwise. */
bool fr_add (struct fraction_t fract1,
             struct fraction_t fract2,
             struct fraction_t *res)
{
  unsigned long member1 ;
  unsigned long member2 ;

  /* Check the integrity of the received fractions. */
  if ((fract1.denom == 0) || (fract2.denom == 0)) return (false) ;
  res->denom = fract1.denom * fract2.denom ;
  member1 = fract1.num * fract2.denom ;
  member2 = fract2.num * fract1.denom ;
  /* If the sign of both fractions is the same, so will be the sign of the
     result. */
  if (fract1.sign == fract2.sign) {
    res->sign = fract1.sign ;
    res->num = member1 + member2 ;
  }
  else {
    /* Signs are differents. Check which scaled denominator is the greatest. */
    if (member1 > member2) {
      res->num = member1 - member2 ;
      res->sign = fract1.sign ;
    }
    else {
      res->num = member2 - member1 ;
      res->sign = fract2.sign ;
    }
  }

  return (true) ;
}


/** Subtract 'fract2' from 'fract1' and set the result in 'fract3' by
    reference passing style. Returns 'true' if the operation is a success,
    'false' otherwise. */
bool fr_sub (struct fraction_t fract1,
             struct fraction_t fract2,
             struct fraction_t *res)
{
  struct fraction_t tmp ;

  /* Check the integrity of the received fractions. */
  if ((fract1.denom == 0) || (fract2.denom == 0)) return (false) ;
  /* Simply add the opposite of 'fract2'. */
  if (fract2.sign == S_pos) tmp.sign = S_neg ;
  else tmp.sign = S_pos ;
  tmp.num = fract2.num ;
  tmp.denom = fract2.denom ;
  return (fr_add (fract1, tmp, res)) ;
}



/** Multiply 'fract1' by 'fract2' and set the result in 'fract3' by reference
    passing style. Returns 'true' if the operation is a success, 'false'
    otherwise. */
bool fr_mul (struct fraction_t fract1,
             struct fraction_t fract2,
             struct fraction_t *res)
{
  /* Check the integrity of the received fractions. */
  if ((fract1.denom == 0) || (fract2.denom == 0)) return (false) ;
  res->num = fract1.num * fract2.num ;
  res->denom = fract1.denom * fract2.denom ;
  if (res->num == 0) res->denom = 1 ;
  /* Compute the sign of the result. */
  if (fract1.sign == fract2.sign) res->sign = S_pos ;
  else res->sign = S_neg ;
  return (true) ;
}


/** Divide 'fract1' by 'fract2' and set the result in 'fract3' by reference
    passing style. Returns 'true' if the operation is a success, 'false'
    otherwise. */
bool fr_div (struct fraction_t fract1,
             struct fraction_t fract2,
             struct fraction_t *res)
{
  /* Check if at least one fraction has a null denominator and if the fraction
     divisor (i.e. 'fract2') is nul. */
  if ((fract1.denom == 0) || (fract2.denom == 0)) return (false) ;
  if (fract2.num == 0) return (false) ;
  res->num = fract1.num * fract2.denom ;
  res->denom = fract1.denom * fract2.num ;
  if (res->num == 0) res->denom = 1 ;
  /* Compute the sign of the result. */
  if (fract1.sign == fract2.sign) res->sign = S_pos ;
  else res->sign = S_neg ;
  return (true) ;
}


/** Compute the irreducible form of 'fract' ands et the result in 'res' by
    reference passing style. Returns 'true' if the operation is a success,
    'false' otherwise. */
bool fr_irr (struct fraction_t fract, struct fraction_t *res)
{
  unsigned long num ;
  unsigned long denom ;
  unsigned long pgcd ;
  unsigned long r ;

  num = fract.num ;
  denom = fract.denom ;
  if (denom == 0) return (false) ;
  /* Simply compute the greatest common divisor. */
  while (denom != 0) {
    r = num % denom ;
    num = denom ;
    denom = r ;
  }
  pgcd = num ;
  /* Here, 'pgcd' containts the GCD of the numerator and denominator.
     Now really reduce the fraction.. */
  res->sign = fract.sign ;
  res->num = fract.num / pgcd ;
  res->denom = fract.denom / pgcd ;
  return (true) ;
}


int main (int argc, char *argv[])
{
  struct fraction_t fr1 = { S_neg, 1, 1 } ;
  struct fraction_t fr2 = { S_pos, 5, 4 } ;
  struct fraction_t fr3 = { S_pos, 12, 8 } ;
  struct fraction_t fr_res ;

  /* Test addition. */
  if (! fr_add (fr1, fr2, &fr_res)) {
    printf ("fr_add: Invalid fraction.\n") ;
  }
  else {
    fr_print (fr1) ; printf (" + ") ;
    fr_print (fr2) ; printf (" = ") ;
    fr_print (fr_res) ; printf ("\n") ;
  }

  /* Test subtraction. */
  if (! fr_sub (fr1, fr2, &fr_res)) {
    printf ("fr_sub: Invalid fraction.\n") ;
  }
  else {
    fr_print (fr1) ; printf (" - ") ;
    fr_print (fr2) ; printf (" = ") ;
    fr_print (fr_res) ; printf ("\n") ;
  }

  /* Test multiplication. */
  if (! fr_mul (fr2, fr3, &fr_res)) {
    printf ("fr_mul: Invalid fraction.\n") ;
  }
  else {
    fr_print (fr2) ; printf (" * ") ;
    fr_print (fr3) ; printf (" = ") ;
    fr_print (fr_res) ; printf ("\n") ;
  }

  /* Test division. */
  if (! fr_div (fr2, fr3, &fr_res)) {
    printf ("fr_div: Invalid fraction.\n") ;
  }
  else {
    fr_print (fr2) ; printf (" / ") ;
    fr_print (fr3) ; printf (" = ") ;
    fr_print (fr_res) ; printf ("\n") ;
  }

  /* Test irreducible form. */
  if (! fr_irr (fr3, &fr_res)) {
    printf ("fr_irr: Invalid fraction.\n") ;
  }
  else {
    fr_print (fr3) ; printf (" reduced = ") ;
    fr_print (fr_res) ; printf ("\n") ;
  }

  return (0) ;
}
