#include "Utilities.cuh"
#include "Bessel.cuh"

/********************************************************/
/* MODIFIED BESSEL FUNCTION CALCULATION DEVICE FUNCTION */
/********************************************************/
template<class T>
__host__ __device__ T bessi0(T x)
{
// -- See paper
// J.M. Blair, "Rational Chebyshev approximations for the modified Bessel functions I_0(x) and I_1(x)", Math. Comput., vol. 28, n. 126, pp. 581-583, Apr. 1974.   

	T num, den, x2;

	x2 = x*x;
	x  = fabs(x);
   
	if (x > static_cast<T>(15.0)) 
	{
		den = static_cast<T>(1.0) / x;
		num =				  static_cast<T>(-4.4979236558557991E+006);
		num = fma2 (num, den, static_cast<T>( 2.7472555659426521E+006));
		num = fma2 (num, den, static_cast<T>(-6.4572046640793153E+005));
		num = fma2 (num, den, static_cast<T>( 8.5476214845610564E+004));
		num = fma2 (num, den, static_cast<T>(-7.1127665397362362E+003));
		num = fma2 (num, den, static_cast<T>( 4.1710918140001479E+002));
		num = fma2 (num, den, static_cast<T>(-1.3787683843558749E+001));
		num = fma2 (num, den, static_cast<T>( 1.1452802345029696E+000));
		num = fma2 (num, den, static_cast<T>( 2.1935487807470277E-001));
		num = fma2 (num, den, static_cast<T>( 9.0727240339987830E-002));
		num = fma2 (num, den, static_cast<T>( 4.4741066428061006E-002));
		num = fma2 (num, den, static_cast<T>( 2.9219412078729436E-002));
		num = fma2 (num, den, static_cast<T>( 2.8050629067165909E-002));
		num = fma2 (num, den, static_cast<T>( 4.9867785050221047E-002));
		num = fma2 (num, den, static_cast<T>( 3.9894228040143265E-001));
		num = num * den;
		den = sqrt (x);
		num = num * den;
		den = exp (static_cast<T>(0.5) * x);  /* prevent premature overflow */
		num = num * den;
		num = num * den;
		return num;
	}
	else
	{
		num = static_cast<T>(-0.27288446572737951578789523409E+010);
		num = fma2 (num, x2, static_cast<T>(-0.6768549084673824894340380223E+009));
		num = fma2 (num, x2, static_cast<T>(-0.4130296432630476829274339869E+008));
		num = fma2 (num, x2, static_cast<T>(-0.11016595146164611763171787004E+007));
		num = fma2 (num, x2, static_cast<T>(-0.1624100026427837007503320319E+005));
		num = fma2 (num, x2, static_cast<T>(-0.1503841142335444405893518061E+003));
		num = fma2 (num, x2, static_cast<T>(-0.947449149975326604416967031E+000));
		num = fma2 (num, x2, static_cast<T>(-0.4287350374762007105516581810E-002));
		num = fma2 (num, x2, static_cast<T>(-0.1447896113298369009581404138E-004));
		num = fma2 (num, x2, static_cast<T>(-0.375114023744978945259642850E-007));
		num = fma2 (num, x2, static_cast<T>(-0.760147559624348256501094832E-010));
		num = fma2 (num, x2, static_cast<T>(-0.121992831543841162565677055E-012));
		num = fma2 (num, x2, static_cast<T>(-0.15587387207852991014838679E-015));
		num = fma2 (num, x2, static_cast<T>(-0.15795544211478823152992269E-018));
		num = fma2 (num, x2, static_cast<T>(-0.1247819710175804058844059E-021));
		num = fma2 (num, x2, static_cast<T>(-0.72585406935875957424755E-025));
		num = fma2 (num, x2, static_cast<T>(-0.28840544803647313855232E-028));      
	  
		den = static_cast<T>(-0.2728844657273795156746641315E+010);
		den = fma2 (den, x2, static_cast<T>(0.5356255851066290475987259E+007));
		den = fma2 (den, x2, static_cast<T>(-0.38305191682802536272760E+004));
		den = fma2 (den, x2, static_cast<T>(0.1E+001));

		return num/den;
	}
}

template __host__ __device__ float  bessi0(float );
template __host__ __device__ double bessi0(double);
