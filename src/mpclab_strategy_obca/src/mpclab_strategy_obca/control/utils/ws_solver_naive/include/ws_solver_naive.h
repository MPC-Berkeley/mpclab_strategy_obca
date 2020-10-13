/*
ws_solver_naive : A fast customized optimization solver.

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

/* Generated by FORCES PRO v4.0.0 on Tuesday, October 13, 2020 at 5:21:48 PM */

#ifndef SOLVER_STDIO_H
#define SOLVER_STDIO_H
#include <stdio.h>
#endif

#ifndef ws_solver_naive_H
#define ws_solver_naive_H

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
typedef double ws_solver_naive_float;

typedef double ws_solver_naiveinterface_float;

/* SOLVER SETTINGS ------------------------------------------------------*/

/* MISRA-C compliance */
#ifndef MISRA_C_ws_solver_naive
#define MISRA_C_ws_solver_naive (0)
#endif

/* restrict code */
#ifndef RESTRICT_CODE_ws_solver_naive
#define RESTRICT_CODE_ws_solver_naive (0)
#endif

/* print level */
#ifndef SET_PRINTLEVEL_ws_solver_naive
#define SET_PRINTLEVEL_ws_solver_naive    (1)
#endif

/* timing */
#ifndef SET_TIMING_ws_solver_naive
#define SET_TIMING_ws_solver_naive    (1)
#endif

/* Numeric Warnings */
/* #define PRINTNUMERICALWARNINGS */

/* maximum number of iterations  */
#define SET_MAXIT_ws_solver_naive			(200)	

/* scaling factor of line search (FTB rule) */
#define SET_FLS_SCALE_ws_solver_naive		(ws_solver_naive_float)(0.99)      

/* maximum number of supported elements in the filter */
#define MAX_FILTER_SIZE_ws_solver_naive	(200) 

/* maximum number of supported elements in the filter */
#define MAX_SOC_IT_ws_solver_naive			(4) 

/* desired relative duality gap */
#define SET_ACC_RDGAP_ws_solver_naive		(ws_solver_naive_float)(0.0001)

/* desired maximum residual on equality constraints */
#define SET_ACC_RESEQ_ws_solver_naive		(ws_solver_naive_float)(1E-06)

/* desired maximum residual on inequality constraints */
#define SET_ACC_RESINEQ_ws_solver_naive	(ws_solver_naive_float)(1E-06)

/* desired maximum violation of complementarity */
#define SET_ACC_KKTCOMPL_ws_solver_naive	(ws_solver_naive_float)(1E-06)


/* RETURN CODES----------------------------------------------------------*/
/* solver has converged within desired accuracy */
#define OPTIMAL_ws_solver_naive      (1)

/* maximum number of iterations has been reached */
#define MAXITREACHED_ws_solver_naive (0)

/* solver has stopped due to a timeout */
#define TIMEOUT_ws_solver_naive   (2)

/* wrong number of inequalities error */
#define INVALID_NUM_INEQ_ERROR_ws_solver_naive  (-4)

/* factorization error */
#define FACTORIZATION_ERROR_ws_solver_naive   (-5)

/* NaN encountered in function evaluations */
#define BADFUNCEVAL_ws_solver_naive  (-6)

/* no progress in method possible */
#define NOPROGRESS_ws_solver_naive   (-7)

/* invalid values in parameters */
#define PARAM_VALUE_ERROR_ws_solver_naive   (-11)

/* too small timeout given */
#define INVALID_TIMEOUT_ws_solver_naive   (-12)

/* licensing error - solver not valid on this machine */
#define LICENSE_ERROR_ws_solver_naive  (-100)

/* PARAMETERS -----------------------------------------------------------*/
/* fill this with data before calling the solver! */
typedef struct
{
    /* vector of size 441 */
    ws_solver_naive_float x0[441];

    /* vector of size 462 */
    ws_solver_naive_float all_parameters[462];


} ws_solver_naive_params;


/* OUTPUTS --------------------------------------------------------------*/
/* the desired variables are put here by the solver */
typedef struct
{
    /* vector of size 21 */
    ws_solver_naive_float x01[21];

    /* vector of size 21 */
    ws_solver_naive_float x02[21];

    /* vector of size 21 */
    ws_solver_naive_float x03[21];

    /* vector of size 21 */
    ws_solver_naive_float x04[21];

    /* vector of size 21 */
    ws_solver_naive_float x05[21];

    /* vector of size 21 */
    ws_solver_naive_float x06[21];

    /* vector of size 21 */
    ws_solver_naive_float x07[21];

    /* vector of size 21 */
    ws_solver_naive_float x08[21];

    /* vector of size 21 */
    ws_solver_naive_float x09[21];

    /* vector of size 21 */
    ws_solver_naive_float x10[21];

    /* vector of size 21 */
    ws_solver_naive_float x11[21];

    /* vector of size 21 */
    ws_solver_naive_float x12[21];

    /* vector of size 21 */
    ws_solver_naive_float x13[21];

    /* vector of size 21 */
    ws_solver_naive_float x14[21];

    /* vector of size 21 */
    ws_solver_naive_float x15[21];

    /* vector of size 21 */
    ws_solver_naive_float x16[21];

    /* vector of size 21 */
    ws_solver_naive_float x17[21];

    /* vector of size 21 */
    ws_solver_naive_float x18[21];

    /* vector of size 21 */
    ws_solver_naive_float x19[21];

    /* vector of size 21 */
    ws_solver_naive_float x20[21];

    /* vector of size 21 */
    ws_solver_naive_float x21[21];


} ws_solver_naive_output;


/* SOLVER INFO ----------------------------------------------------------*/
/* diagnostic data from last interior point step */
typedef struct
{
    /* iteration number */
    solver_int32_default it;

	/* number of iterations needed to optimality (branch-and-bound) */
	solver_int32_default it2opt;
	
    /* inf-norm of equality constraint residuals */
    ws_solver_naive_float res_eq;
	
    /* inf-norm of inequality constraint residuals */
    ws_solver_naive_float res_ineq;

	/* norm of stationarity condition */
    ws_solver_naive_float rsnorm;

	/* max of all complementarity violations */
    ws_solver_naive_float rcompnorm;

    /* primal objective */
    ws_solver_naive_float pobj;	
	
    /* dual objective */
    ws_solver_naive_float dobj;	

    /* duality gap := pobj - dobj */
    ws_solver_naive_float dgap;		
	
    /* relative duality gap := |dgap / pobj | */
    ws_solver_naive_float rdgap;		

    /* duality measure */
    ws_solver_naive_float mu;

	/* duality measure (after affine step) */
    ws_solver_naive_float mu_aff;
	
    /* centering parameter */
    ws_solver_naive_float sigma;
	
    /* number of backtracking line search steps (affine direction) */
    solver_int32_default lsit_aff;
    
    /* number of backtracking line search steps (combined direction) */
    solver_int32_default lsit_cc;
    
    /* step size (affine direction) */
    ws_solver_naive_float step_aff;
    
    /* step size (combined direction) */
    ws_solver_naive_float step_cc;    

	/* solvertime */
	ws_solver_naive_float solvetime;   

	/* time spent in function evaluations */
	ws_solver_naive_float fevalstime;  


} ws_solver_naive_info;







/* SOLVER FUNCTION DEFINITION -------------------------------------------*/
/* Time of Solver Generation: (UTC) Tuesday, October 13, 2020 5:21:50 PM */
/* User License expires on: (UTC) Friday, February 26, 2021 10:00:00 PM (approx.) (at the time of code generation) */
/* Solver Static License expires on: (UTC) Friday, February 26, 2021 10:00:00 PM (approx.) */
/* Solver Generation Request Id: 7fde1963-62b6-40a2-8e96-c8611e58ba68 */
/* examine exitflag before using the result! */
#ifdef __cplusplus
extern "C" {
#endif		

typedef void (*ws_solver_naive_extfunc)(ws_solver_naive_float* x, ws_solver_naive_float* y, ws_solver_naive_float* lambda, ws_solver_naive_float* params, ws_solver_naive_float* pobj, ws_solver_naive_float* g, ws_solver_naive_float* c, ws_solver_naive_float* Jeq, ws_solver_naive_float* h, ws_solver_naive_float* Jineq, ws_solver_naive_float* H, solver_int32_default stage, solver_int32_default iterations);

extern solver_int32_default ws_solver_naive_solve(ws_solver_naive_params *params, ws_solver_naive_output *output, ws_solver_naive_info *info, FILE *fs, ws_solver_naive_extfunc evalextfunctions_ws_solver_naive);	





#ifdef __cplusplus
}
#endif

#endif
