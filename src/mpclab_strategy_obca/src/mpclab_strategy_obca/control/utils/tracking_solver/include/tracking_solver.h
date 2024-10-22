/*
tracking_solver : A fast customized optimization solver.

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

/* Generated by FORCES PRO v4.0.0 on Friday, October 16, 2020 at 5:49:10 PM */

#ifndef SOLVER_STDIO_H
#define SOLVER_STDIO_H
#include <stdio.h>
#endif

#ifndef tracking_solver_H
#define tracking_solver_H

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
typedef double tracking_solver_float;

typedef double tracking_solverinterface_float;

/* SOLVER SETTINGS ------------------------------------------------------*/

/* MISRA-C compliance */
#ifndef MISRA_C_tracking_solver
#define MISRA_C_tracking_solver (0)
#endif

/* restrict code */
#ifndef RESTRICT_CODE_tracking_solver
#define RESTRICT_CODE_tracking_solver (0)
#endif

/* print level */
#ifndef SET_PRINTLEVEL_tracking_solver
#define SET_PRINTLEVEL_tracking_solver    (1)
#endif

/* timing */
#ifndef SET_TIMING_tracking_solver
#define SET_TIMING_tracking_solver    (1)
#endif

/* Numeric Warnings */
/* #define PRINTNUMERICALWARNINGS */

/* maximum number of iterations  */
#define SET_MAXIT_tracking_solver			(200)	

/* scaling factor of line search (FTB rule) */
#define SET_FLS_SCALE_tracking_solver		(tracking_solver_float)(0.99)      

/* maximum number of supported elements in the filter */
#define MAX_FILTER_SIZE_tracking_solver	(200) 

/* maximum number of supported elements in the filter */
#define MAX_SOC_IT_tracking_solver			(4) 

/* desired relative duality gap */
#define SET_ACC_RDGAP_tracking_solver		(tracking_solver_float)(0.001)

/* desired maximum residual on equality constraints */
#define SET_ACC_RESEQ_tracking_solver		(tracking_solver_float)(0.001)

/* desired maximum residual on inequality constraints */
#define SET_ACC_RESINEQ_tracking_solver	(tracking_solver_float)(0.001)

/* desired maximum violation of complementarity */
#define SET_ACC_KKTCOMPL_tracking_solver	(tracking_solver_float)(0.001)


/* RETURN CODES----------------------------------------------------------*/
/* solver has converged within desired accuracy */
#define OPTIMAL_tracking_solver      (1)

/* maximum number of iterations has been reached */
#define MAXITREACHED_tracking_solver (0)

/* solver has stopped due to a timeout */
#define TIMEOUT_tracking_solver   (2)

/* wrong number of inequalities error */
#define INVALID_NUM_INEQ_ERROR_tracking_solver  (-4)

/* factorization error */
#define FACTORIZATION_ERROR_tracking_solver   (-5)

/* NaN encountered in function evaluations */
#define BADFUNCEVAL_tracking_solver  (-6)

/* no progress in method possible */
#define NOPROGRESS_tracking_solver   (-7)

/* invalid values in parameters */
#define PARAM_VALUE_ERROR_tracking_solver   (-11)

/* too small timeout given */
#define INVALID_TIMEOUT_tracking_solver   (-12)

/* licensing error - solver not valid on this machine */
#define LICENSE_ERROR_tracking_solver  (-100)

/* PARAMETERS -----------------------------------------------------------*/
/* fill this with data before calling the solver! */
typedef struct
{
    /* vector of size 398 */
    tracking_solver_float lb[398];

    /* vector of size 398 */
    tracking_solver_float ub[398];

    /* vector of size 404 */
    tracking_solver_float x0[404];

    /* vector of size 6 */
    tracking_solver_float xinit[6];

    /* vector of size 608 */
    tracking_solver_float all_parameters[608];


} tracking_solver_params;


/* OUTPUTS --------------------------------------------------------------*/
/* the desired variables are put here by the solver */
typedef struct
{
    /* vector of size 8 */
    tracking_solver_float x01[8];

    /* vector of size 8 */
    tracking_solver_float x02[8];

    /* vector of size 8 */
    tracking_solver_float x03[8];

    /* vector of size 8 */
    tracking_solver_float x04[8];

    /* vector of size 8 */
    tracking_solver_float x05[8];

    /* vector of size 8 */
    tracking_solver_float x06[8];

    /* vector of size 8 */
    tracking_solver_float x07[8];

    /* vector of size 8 */
    tracking_solver_float x08[8];

    /* vector of size 8 */
    tracking_solver_float x09[8];

    /* vector of size 8 */
    tracking_solver_float x10[8];

    /* vector of size 8 */
    tracking_solver_float x11[8];

    /* vector of size 8 */
    tracking_solver_float x12[8];

    /* vector of size 8 */
    tracking_solver_float x13[8];

    /* vector of size 8 */
    tracking_solver_float x14[8];

    /* vector of size 8 */
    tracking_solver_float x15[8];

    /* vector of size 8 */
    tracking_solver_float x16[8];

    /* vector of size 8 */
    tracking_solver_float x17[8];

    /* vector of size 8 */
    tracking_solver_float x18[8];

    /* vector of size 8 */
    tracking_solver_float x19[8];

    /* vector of size 8 */
    tracking_solver_float x20[8];

    /* vector of size 8 */
    tracking_solver_float x21[8];

    /* vector of size 8 */
    tracking_solver_float x22[8];

    /* vector of size 8 */
    tracking_solver_float x23[8];

    /* vector of size 8 */
    tracking_solver_float x24[8];

    /* vector of size 8 */
    tracking_solver_float x25[8];

    /* vector of size 8 */
    tracking_solver_float x26[8];

    /* vector of size 8 */
    tracking_solver_float x27[8];

    /* vector of size 8 */
    tracking_solver_float x28[8];

    /* vector of size 8 */
    tracking_solver_float x29[8];

    /* vector of size 8 */
    tracking_solver_float x30[8];

    /* vector of size 8 */
    tracking_solver_float x31[8];

    /* vector of size 8 */
    tracking_solver_float x32[8];

    /* vector of size 8 */
    tracking_solver_float x33[8];

    /* vector of size 8 */
    tracking_solver_float x34[8];

    /* vector of size 8 */
    tracking_solver_float x35[8];

    /* vector of size 8 */
    tracking_solver_float x36[8];

    /* vector of size 8 */
    tracking_solver_float x37[8];

    /* vector of size 8 */
    tracking_solver_float x38[8];

    /* vector of size 8 */
    tracking_solver_float x39[8];

    /* vector of size 8 */
    tracking_solver_float x40[8];

    /* vector of size 8 */
    tracking_solver_float x41[8];

    /* vector of size 8 */
    tracking_solver_float x42[8];

    /* vector of size 8 */
    tracking_solver_float x43[8];

    /* vector of size 8 */
    tracking_solver_float x44[8];

    /* vector of size 8 */
    tracking_solver_float x45[8];

    /* vector of size 8 */
    tracking_solver_float x46[8];

    /* vector of size 8 */
    tracking_solver_float x47[8];

    /* vector of size 8 */
    tracking_solver_float x48[8];

    /* vector of size 8 */
    tracking_solver_float x49[8];

    /* vector of size 8 */
    tracking_solver_float x50[8];

    /* vector of size 4 */
    tracking_solver_float x51[4];


} tracking_solver_output;


/* SOLVER INFO ----------------------------------------------------------*/
/* diagnostic data from last interior point step */
typedef struct
{
    /* iteration number */
    solver_int32_default it;

	/* number of iterations needed to optimality (branch-and-bound) */
	solver_int32_default it2opt;
	
    /* inf-norm of equality constraint residuals */
    tracking_solver_float res_eq;
	
    /* inf-norm of inequality constraint residuals */
    tracking_solver_float res_ineq;

	/* norm of stationarity condition */
    tracking_solver_float rsnorm;

	/* max of all complementarity violations */
    tracking_solver_float rcompnorm;

    /* primal objective */
    tracking_solver_float pobj;	
	
    /* dual objective */
    tracking_solver_float dobj;	

    /* duality gap := pobj - dobj */
    tracking_solver_float dgap;		
	
    /* relative duality gap := |dgap / pobj | */
    tracking_solver_float rdgap;		

    /* duality measure */
    tracking_solver_float mu;

	/* duality measure (after affine step) */
    tracking_solver_float mu_aff;
	
    /* centering parameter */
    tracking_solver_float sigma;
	
    /* number of backtracking line search steps (affine direction) */
    solver_int32_default lsit_aff;
    
    /* number of backtracking line search steps (combined direction) */
    solver_int32_default lsit_cc;
    
    /* step size (affine direction) */
    tracking_solver_float step_aff;
    
    /* step size (combined direction) */
    tracking_solver_float step_cc;    

	/* solvertime */
	tracking_solver_float solvetime;   

	/* time spent in function evaluations */
	tracking_solver_float fevalstime;  


} tracking_solver_info;







/* SOLVER FUNCTION DEFINITION -------------------------------------------*/
/* Time of Solver Generation: (UTC) Friday, October 16, 2020 5:49:13 PM */
/* User License expires on: (UTC) Friday, February 26, 2021 10:00:00 PM (approx.) (at the time of code generation) */
/* Solver Static License expires on: (UTC) Friday, February 26, 2021 10:00:00 PM (approx.) */
/* Solver Generation Request Id: 8c6e5085-6fc6-47e4-a720-6a006347ebd4 */
/* examine exitflag before using the result! */
#ifdef __cplusplus
extern "C" {
#endif		

typedef void (*tracking_solver_extfunc)(tracking_solver_float* x, tracking_solver_float* y, tracking_solver_float* lambda, tracking_solver_float* params, tracking_solver_float* pobj, tracking_solver_float* g, tracking_solver_float* c, tracking_solver_float* Jeq, tracking_solver_float* h, tracking_solver_float* Jineq, tracking_solver_float* H, solver_int32_default stage, solver_int32_default iterations);

extern solver_int32_default tracking_solver_solve(tracking_solver_params *params, tracking_solver_output *output, tracking_solver_info *info, FILE *fs, tracking_solver_extfunc evalextfunctions_tracking_solver);	





#ifdef __cplusplus
}
#endif

#endif
