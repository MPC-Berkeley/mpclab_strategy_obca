/*
opt_solver_naive : A fast customized optimization solver.

Copyright (C) 2013-2020 EMBOTECH AG [info@embotech.com]. All rights reserved.


This software is intended for simulation and testing purposes only. 
Use of this software for any commercial purpose is prohibited.

This program is distributed in the hope that it will be useful.
EMBOTECH makes NO WARRANTIES with respect to the use of the software 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. 

EMBOTECH shall not have any liability for any damage arising from the use
of the software.

This Agreement shall exclusively be governed by and interpreted in 
accordance with the laws of Switzerland, excluding its principles
of conflict of laws. The Courts of Zurich-City shall have exclusive 
jurisdiction in case of any dispute.

*/

/* Generated by FORCES PRO v4.0.0 on Wednesday, October 14, 2020 at 1:04:15 AM */

#ifndef SOLVER_STDIO_H
#define SOLVER_STDIO_H
#include <stdio.h>
#endif

#ifndef opt_solver_naive_H
#define opt_solver_naive_H

#ifndef SOLVER_STANDARD_TYPES
#define SOLVER_STANDARD_TYPES

typedef signed char solver_int8_signed;
typedef unsigned char solver_int8_unsigned;
typedef char solver_int8_default;
typedef signed short int solver_int16_signed;
typedef unsigned short int solver_int16_unsigned;
typedef short int solver_int16_default;
typedef signed int solver_int32_signed;
typedef unsigned int solver_int32_unsigned;
typedef int solver_int32_default;
typedef signed long long int solver_int64_signed;
typedef unsigned long long int solver_int64_unsigned;
typedef long long int solver_int64_default;

#endif


/* DATA TYPE ------------------------------------------------------------*/
typedef double opt_solver_naive_float;

typedef double opt_solver_naiveinterface_float;

/* SOLVER SETTINGS ------------------------------------------------------*/

/* MISRA-C compliance */
#ifndef MISRA_C_opt_solver_naive
#define MISRA_C_opt_solver_naive (0)
#endif

/* restrict code */
#ifndef RESTRICT_CODE_opt_solver_naive
#define RESTRICT_CODE_opt_solver_naive (0)
#endif

/* print level */
#ifndef SET_PRINTLEVEL_opt_solver_naive
#define SET_PRINTLEVEL_opt_solver_naive    (1)
#endif

/* timing */
#ifndef SET_TIMING_opt_solver_naive
#define SET_TIMING_opt_solver_naive    (1)
#endif

/* Numeric Warnings */
/* #define PRINTNUMERICALWARNINGS */

/* maximum number of iterations  */
#define SET_MAXIT_opt_solver_naive			(200)	

/* scaling factor of line search (FTB rule) */
#define SET_FLS_SCALE_opt_solver_naive		(opt_solver_naive_float)(0.99)      

/* maximum number of supported elements in the filter */
#define MAX_FILTER_SIZE_opt_solver_naive	(200) 

/* maximum number of supported elements in the filter */
#define MAX_SOC_IT_opt_solver_naive			(4) 

/* desired relative duality gap */
#define SET_ACC_RDGAP_opt_solver_naive		(opt_solver_naive_float)(0.001)

/* desired maximum residual on equality constraints */
#define SET_ACC_RESEQ_opt_solver_naive		(opt_solver_naive_float)(0.001)

/* desired maximum residual on inequality constraints */
#define SET_ACC_RESINEQ_opt_solver_naive	(opt_solver_naive_float)(0.001)

/* desired maximum violation of complementarity */
#define SET_ACC_KKTCOMPL_opt_solver_naive	(opt_solver_naive_float)(0.001)


/* RETURN CODES----------------------------------------------------------*/
/* solver has converged within desired accuracy */
#define OPTIMAL_opt_solver_naive      (1)

/* maximum number of iterations has been reached */
#define MAXITREACHED_opt_solver_naive (0)

/* solver has stopped due to a timeout */
#define TIMEOUT_opt_solver_naive   (2)

/* wrong number of inequalities error */
#define INVALID_NUM_INEQ_ERROR_opt_solver_naive  (-4)

/* factorization error */
#define FACTORIZATION_ERROR_opt_solver_naive   (-5)

/* NaN encountered in function evaluations */
#define BADFUNCEVAL_opt_solver_naive  (-6)

/* no progress in method possible */
#define NOPROGRESS_opt_solver_naive   (-7)

/* invalid values in parameters */
#define PARAM_VALUE_ERROR_opt_solver_naive   (-11)

/* too small timeout given */
#define INVALID_TIMEOUT_opt_solver_naive   (-12)

/* licensing error - solver not valid on this machine */
#define LICENSE_ERROR_opt_solver_naive  (-100)

/* PARAMETERS -----------------------------------------------------------*/
/* fill this with data before calling the solver! */
typedef struct
{
    /* vector of size 542 */
    opt_solver_naive_float x0[542];

    /* vector of size 6 */
    opt_solver_naive_float xinit[6];

    /* vector of size 462 */
    opt_solver_naive_float all_parameters[462];


} opt_solver_naive_params;


/* OUTPUTS --------------------------------------------------------------*/
/* the desired variables are put here by the solver */
typedef struct
{
    /* vector of size 26 */
    opt_solver_naive_float x01[26];

    /* vector of size 26 */
    opt_solver_naive_float x02[26];

    /* vector of size 26 */
    opt_solver_naive_float x03[26];

    /* vector of size 26 */
    opt_solver_naive_float x04[26];

    /* vector of size 26 */
    opt_solver_naive_float x05[26];

    /* vector of size 26 */
    opt_solver_naive_float x06[26];

    /* vector of size 26 */
    opt_solver_naive_float x07[26];

    /* vector of size 26 */
    opt_solver_naive_float x08[26];

    /* vector of size 26 */
    opt_solver_naive_float x09[26];

    /* vector of size 26 */
    opt_solver_naive_float x10[26];

    /* vector of size 26 */
    opt_solver_naive_float x11[26];

    /* vector of size 26 */
    opt_solver_naive_float x12[26];

    /* vector of size 26 */
    opt_solver_naive_float x13[26];

    /* vector of size 26 */
    opt_solver_naive_float x14[26];

    /* vector of size 26 */
    opt_solver_naive_float x15[26];

    /* vector of size 26 */
    opt_solver_naive_float x16[26];

    /* vector of size 26 */
    opt_solver_naive_float x17[26];

    /* vector of size 26 */
    opt_solver_naive_float x18[26];

    /* vector of size 26 */
    opt_solver_naive_float x19[26];

    /* vector of size 26 */
    opt_solver_naive_float x20[26];

    /* vector of size 22 */
    opt_solver_naive_float x21[22];


} opt_solver_naive_output;


/* SOLVER INFO ----------------------------------------------------------*/
/* diagnostic data from last interior point step */
typedef struct
{
    /* iteration number */
    solver_int32_default it;

	/* number of iterations needed to optimality (branch-and-bound) */
	solver_int32_default it2opt;
	
    /* inf-norm of equality constraint residuals */
    opt_solver_naive_float res_eq;
	
    /* inf-norm of inequality constraint residuals */
    opt_solver_naive_float res_ineq;

	/* norm of stationarity condition */
    opt_solver_naive_float rsnorm;

	/* max of all complementarity violations */
    opt_solver_naive_float rcompnorm;

    /* primal objective */
    opt_solver_naive_float pobj;	
	
    /* dual objective */
    opt_solver_naive_float dobj;	

    /* duality gap := pobj - dobj */
    opt_solver_naive_float dgap;		
	
    /* relative duality gap := |dgap / pobj | */
    opt_solver_naive_float rdgap;		

    /* duality measure */
    opt_solver_naive_float mu;

	/* duality measure (after affine step) */
    opt_solver_naive_float mu_aff;
	
    /* centering parameter */
    opt_solver_naive_float sigma;
	
    /* number of backtracking line search steps (affine direction) */
    solver_int32_default lsit_aff;
    
    /* number of backtracking line search steps (combined direction) */
    solver_int32_default lsit_cc;
    
    /* step size (affine direction) */
    opt_solver_naive_float step_aff;
    
    /* step size (combined direction) */
    opt_solver_naive_float step_cc;    

	/* solvertime */
	opt_solver_naive_float solvetime;   

	/* time spent in function evaluations */
	opt_solver_naive_float fevalstime;  


} opt_solver_naive_info;







/* SOLVER FUNCTION DEFINITION -------------------------------------------*/
/* Time of Solver Generation: (UTC) Wednesday, October 14, 2020 1:04:17 AM */
/* User License expires on: (UTC) Friday, February 26, 2021 10:00:00 PM (approx.) (at the time of code generation) */
/* Solver Static License expires on: (UTC) Friday, February 26, 2021 10:00:00 PM (approx.) */
/* Solver Generation Request Id: b5ded74b-9c77-4817-8195-a2efcdaa1e7e */
/* examine exitflag before using the result! */
#ifdef __cplusplus
extern "C" {
#endif		

typedef void (*opt_solver_naive_extfunc)(opt_solver_naive_float* x, opt_solver_naive_float* y, opt_solver_naive_float* lambda, opt_solver_naive_float* params, opt_solver_naive_float* pobj, opt_solver_naive_float* g, opt_solver_naive_float* c, opt_solver_naive_float* Jeq, opt_solver_naive_float* h, opt_solver_naive_float* Jineq, opt_solver_naive_float* H, solver_int32_default stage, solver_int32_default iterations);

extern solver_int32_default opt_solver_naive_solve(opt_solver_naive_params *params, opt_solver_naive_output *output, opt_solver_naive_info *info, FILE *fs, opt_solver_naive_extfunc evalextfunctions_opt_solver_naive);	





#ifdef __cplusplus
}
#endif

#endif
