################################################################################
"""

ks_solver.py

Author: Matthew Holland

Dependencies:

    ./lib/distrubuted*


This python file contains the objects required to perform a solve on the
    Kuramoto-Sivashinsky equation (KS). This corresponds to the effort on
    the class project for ME-5653.

Changelog:

Version     Date        Description

0.0         2024/12/06  Original version




"""
###############################################################################

############################################################################
#
# Import required modules
#
############################################################################

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spsp
import scipy.sparse as spsr

# Add the directory containing your module to sys.path
module_path = os.path.abspath( r"..\code\lib" )
sys.path.append(module_path)

from distributedObjects import *
from distributedFunctions import *

############################################################################
#
# KS Equation object
#
############################################################################

class KS:
    """
    This object contains the necessary data and functions to solve the
        Kuramoto-Sivashinsky equations

    """

    def __init__( self , x , u_0 , t_bounds , dt , alpha=-1e-6 , beta=0.0 , gamma=1e-6 ):
        """
        This method initialized the KS equation object to set up the solver.

        Args:
            x [float]:      The spatial domain to calculate the KS equation
                                over.

            u_0 [float]:    The initial values of the function to initialize
                                the KS equation.

            t_bounds (float):   The bounds of time to solve over.

            dt (float):     The time step size for the solution.

            alpha (float, optional):  The value for the \alpha coefficient. 
                                            Defaults to 1.0.

            beta (float, optional): The value for the \beta coefficient. 
                                            Defaults to 0.0.

            gamma (float, optional):    The value for the \gamma coefficient. 
                                            Defaults to 1.0.

        Attributes:
            x   <-  x

            u_0 <-  u_0

            t_st (float):   The starting time for the KS solve. min(t_bounds)

            t_sp (float):   The end time for the KS solve. max(t_bounds)

            dt  <-  dt

            alpha   <-  alpha
            
            beta    <-  beta

            gamma   <-  gamma

        """

        self.x = x
        self.u_0 = u_0
        self.dt = dt
        self.t_st = np.min( t_bounds )
        self.t_sp = np.max( t_bounds )

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.t = np.arange( self.t_st , self.t_sp , self.dt )
        self.dx = np.mean( np.gradient( self.x ) )

        self.Nx = len( self.x )
        self.Nt = len( self.t )

        self.u = np.zeros(  ( self.Nt , self.Nx ) )
        self.u[0,:] = u_0
        self.v = np.zeros_like( self.u )
        self.f = np.zeros_like( self.u )

        self.Re_cell = np.max( np.abs( self.u ) ) * self.dx / -self.alpha
        self.KS_cell = np.max( np.abs( self.u ) ) * ( self.dx ** 3 ) / self.gamma

    def solve( cls , n_xOrder=4 , n_tOrder=4 , bc_u=(0,0) , bc_dudx=(0,0) , bc_d2udx2=(None,None) , bc_d3udx3=(None,None) , bc_d4udx4=(None,None) , bc_xOrder=1 , zero_tol=1e-12 ):
        """
        Solve the KS equation as initialized.

        The solver equation takes the form:

        D<u_k+1>=A<v_k>+B<u_k>+E<e>

            where v = (u^2/2) and E<e> represents the boundary condition 
                solution

        Args:
            n_xOrder (int, optional): The spatial order of accuracy. 
                                        Defaults to 4.
            n_tOrder (int, optional): The time order of accuracy. The input
                                        values correspond to:
                                        
                                    - 1: Euler time stepping

                                    - 2:    NOPE

                                    - 3:    NOPE

                                    - 4: Runge-Kutta-4 time stepping
                                        
                                        Defaults to 4.

        """
        #
        # Set up time stepping parameters
        #
        cls.f = np.zeros_like( cls.u )
        cls.phi = np.zeros_like( cls.u )
        if n_tOrder==1:
            print("Eulerian time stepping selected")
        elif n_tOrder==4:
            print("RK4 time stepping selected.")
            cls.R_n = np.zeros_like( cls.u )
            cls.R_1 = np.zeros_like( cls.u )
            cls.R_2 = np.zeros_like( cls.u )
            cls.R_3 = np.zeros_like( cls.u )
            cls.u_1 = np.zeros_like( cls.u )
            cls.u_2 = np.zeros_like( cls.u )
            cls.u_3 = np.zeros_like( cls.u )
            cls.v_1 = np.zeros_like( cls.u )
            cls.v_2 = np.zeros_like( cls.u )
            cls.v_3 = np.zeros_like( cls.u )
            cls.f_R = np.zeros_like( cls.u )
            cls.phi_1 = np.zeros_like( cls.u )
            cls.phi_2 = np.zeros_like( cls.u )
            cls.phi_3 = np.zeros_like( cls.u )

        #
        # Calculate the matrix for advective term
        #
        cls.numgradient_advect = numericalGradient( 1 , ( n_xOrder - n_xOrder//2 , n_xOrder//2 ) )
        cls.numgradient_advect.formMatrix( cls.Nx )
        cls.A_advect = cls.numgradient_advect.gradientMatrix / cls.dx
        # Change boundary rows
        cls.A_advect = cls.A_advect.tolil()
        cls.numgradient_LHS_advect = numericalGradient( 1 , ( 0 , n_xOrder ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.A_advect[i,:]=0
            cls.A_advect[i,i:i+n_xOrder+1] = cls.numgradient_LHS_advect.coeffs / cls.dx
        cls.numgradient_RHS_advect = numericalGradient( 1 , ( n_xOrder , 0 ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.A_advect[-i-1,:]=0
            if i>0:
                cls.A_advect[-i-1,-(i+n_xOrder+1):-i] = cls.numgradient_RHS_advect.coeffs / cls.dx
            else:
                cls.A_advect[-i-1,-(n_xOrder+1):] = cls.numgradient_RHS_advect.coeffs / cls.dx
        # Fix float zeros
        cls.A_advect[np.abs(cls.A_advect)*cls.dx<=zero_tol]=0
        cls.A_advect = cls.A_advect.todia()

        #
        # Calculate the matrix for diffusive term
        #
        cls.numgradient_diffuse = numericalGradient( 2 , ( n_xOrder - n_xOrder//2 , n_xOrder//2 ) )
        cls.numgradient_diffuse.formMatrix( cls.Nx )
        cls.B_diffuse = cls.numgradient_diffuse.gradientMatrix / ( cls.dx ** 2)
        # Change boundary rows
        cls.B_diffuse = cls.B_diffuse.tolil()
        cls.numgradient_LHS_diffuse = numericalGradient( 2 , ( 0 , n_xOrder ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.B_diffuse[i,:]=0
            cls.B_diffuse[i,i:i+n_xOrder+1] = cls.numgradient_LHS_diffuse.coeffs / ( cls.dx ** 2)
        cls.numgradient_RHS_diffuse = numericalGradient( 2 , ( n_xOrder , 0 ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.B_diffuse[-i-1,:]=0
            if i>0:
                cls.B_diffuse[-i-1,-(i+n_xOrder+1):-i] = cls.numgradient_RHS_diffuse.coeffs / ( cls.dx ** 2)
            else:
                cls.B_diffuse[-i-1,-(n_xOrder+1):] = cls.numgradient_RHS_diffuse.coeffs / ( cls.dx ** 2)
        cls.B_diffuse = cls.B_diffuse.todia()  

        #
        # Calculate the matrix for the 3rd derivative term
        #
        cls.numgradient_third = numericalGradient( 3 , ( n_xOrder - n_xOrder//2 , n_xOrder//2 ) )
        cls.numgradient_third.formMatrix( cls.Nx )
        cls.B_third = cls.numgradient_third.gradientMatrix / ( cls.dx ** 3 )
        # Change boundary rows
        cls.B_third = cls.B_third.tolil()
        cls.numgradient_LHS_third = numericalGradient( 3 , ( 0 , n_xOrder ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.B_third[i,:]=0
            cls.B_third[i,i:i+n_xOrder+1] = cls.numgradient_LHS_third.coeffs / ( cls.dx ** 3 )
        cls.numgradient_RHS_third = numericalGradient( 3 , ( n_xOrder , 0 ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.B_third[-i-1,:]=0
            if i>0:
                cls.B_third[-i-1,-(i+n_xOrder+1):-i] = cls.numgradient_RHS_third.coeffs / ( cls.dx ** 3 )
            else:
                cls.B_third[-i-1,-(n_xOrder+1):] = cls.numgradient_RHS_third.coeffs / ( cls.dx ** 3 )
        cls.B_third = cls.B_third.todia() 

        #
        # Calculate the matrix for the 4th derivative term
        #
        cls.numgradient_fourth = numericalGradient( 4 , ( n_xOrder - n_xOrder//2 , n_xOrder//2 ) )
        cls.numgradient_fourth.formMatrix( cls.Nx )
        cls.B_fourth = cls.numgradient_fourth.gradientMatrix / ( cls.dx ** 4 )
        # Change boundary rows
        cls.B_fourth = cls.B_fourth.tolil()
        cls.numgradient_LHS_fourth = numericalGradient( 4 , ( 0 , n_xOrder ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.B_fourth[i,:]=0
            cls.B_fourth[i,i:i+n_xOrder+1] = cls.numgradient_LHS_fourth.coeffs / ( cls.dx ** 4 )
        cls.numgradient_RHS_fourth = numericalGradient( 4 , ( n_xOrder , 0 ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.B_fourth[-i-1,:]=0
            if i>0:
                cls.B_fourth[-i-1,-(i+n_xOrder+1):-i] = cls.numgradient_RHS_fourth.coeffs / ( cls.dx ** 4 )
            else:
                cls.B_fourth[-i-1,-(n_xOrder+1):] = cls.numgradient_RHS_fourth.coeffs / ( cls.dx ** 4 )
        cls.B_fourth = cls.B_fourth.todia()

        #
        # Combine matrices
        #  
        cls.A = -cls.A_advect
        cls.B = -cls.alpha * cls.B_diffuse - cls.beta * cls.B_third - cls.gamma * cls.B_fourth

        #
        # Create matrix for the LHS
        #
        cls.D = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        #
        # Create matrix for the Neumann BC's
        #
        cls.E = spsr.dia_matrix( ( np.zeros( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )
        cls.e = np.zeros( cls.Nx )

        #
        # Place boundary conditions
        #
        bc_count=0
        bc_LHS_count=0
        bc_RHS_count=0
        cls.A = cls.A.tolil()
        cls.B = cls.B.tolil()
        cls.D = cls.D.tolil()
        cls.E = cls.E.tolil()
        # Neumann boundary condition
        if bc_u[0]:
            cls.e[0] = bc_u[0]
        if bc_u[-1]:
            cls.e[-1] = bc_u[-1]
        for i , bc in enumerate( bc_u ):
            if bc or bc==0:
                bc_count += 1

                if i==0:
                    cls.A[0,:] = 0
                    cls.B[0,:] = 0
                    cls.D[0,:] = 0
                    cls.D[0,0] = 1
                    cls.E[0,0] = 1
                    bc_LHS_count += 1

                if i==len(bc_u)-1:
                    cls.A[-1,:] = 0
                    cls.B[-1,:] = 0
                    cls.D[-1,:] = 0
                    cls.D[-1,-1] = 1
                    cls.E[-1,-1] = 1
                    bc_RHS_count += 1
        # Dirichlet boundary condition
        if bc_dudx[0]:
            cls.e[0] = bc_dudx[0]
        if bc_dudx[-1]:
            cls.e[-1] = bc_dudx[-1]
        for i , bc in enumerate( bc_dudx ):
            if bc or bc==0:
                bc_count += 1

                if i==0:
                    numgradient_BC = numericalGradient( 1 , ( 0 , bc_xOrder ) )
                    cls.A[bc_LHS_count,:] = 0
                    cls.B[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:bc_xOrder+1] = numgradient_BC.coeffs
                    cls.E[bc_LHS_count,0] = 1
                    bc_LHS_count += 1

                if i==len(bc_u)-1:
                    numgradient_BC = numericalGradient( 1 , ( bc_xOrder , 0 ) )
                    cls.A[-1-bc_RHS_count,:] = 0
                    cls.B[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,-(bc_xOrder+1):] = numgradient_BC.coeffs
                    cls.E[-1-bc_RHS_count,-1] = 1
                    bc_RHS_count += 1
        # Diffusion boundary condition
        if bc_d2udx2[0]:
            cls.e[0] = bc_d2udx2[0]
        if bc_d2udx2[-1]:
            cls.e[-1] = bc_d2udx2[-1]
        for i , bc in enumerate( bc_d2udx2 ):
            if bc or bc==0:
                bc_count += 1

                if i==0:
                    numgradient_BC = numericalGradient( 2 , ( 0 , bc_xOrder ) )
                    cls.A[bc_LHS_count,:] = 0
                    cls.B[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:bc_xOrder+1] = numgradient_BC.coeffs
                    cls.E[bc_LHS_count,0] = 1
                    bc_LHS_count += 1

                if i==len(bc_u)-1:
                    numgradient_BC = numericalGradient( 2 , ( bc_xOrder , 0 ) )
                    cls.A[-1-bc_RHS_count,:] = 0
                    cls.B[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,-(bc_xOrder+1):] = numgradient_BC.coeffs
                    cls.E[-1-bc_RHS_count,-1] = 1
                    bc_RHS_count += 1
        # Third order derivative boundary condition
        if bc_d3udx3[0]:
            cls.e[0] = bc_d3udx3[0]
        if bc_d3udx3[-1]:
            cls.e[-1] = bc_d3udx3[-1]
        for i , bc in enumerate( bc_d3udx3 ):
            if bc or bc==0:
                bc_count += 1

                if i==0:
                    numgradient_BC = numericalGradient( 3 , ( 0 , bc_xOrder ) )
                    cls.A[bc_LHS_count,:] = 0
                    cls.B[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:bc_xOrder+1] = numgradient_BC.coeffs
                    cls.E[bc_LHS_count,0] = 1
                    bc_LHS_count += 1

                if i==len(bc_u)-1:
                    numgradient_BC = numericalGradient( 3 , ( bc_xOrder , 0 ) )
                    cls.A[-1-bc_RHS_count,:] = 0
                    cls.B[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,-(bc_xOrder+1):] = numgradient_BC.coeffs
                    cls.E[-1-bc_RHS_count,-1] = 1
                    bc_RHS_count += 1
        # Fourth order derivative boundary condition
        if bc_d4udx4[0]:
            cls.e[0] = bc_d4udx4[0]
        if bc_d4udx4[-1]:
            cls.e[-1] = bc_d4udx4[-1]
        for i , bc in enumerate( bc_d4udx4 ):
            if bc or bc==0:
                bc_count += 1

                if i==0:
                    numgradient_BC = numericalGradient( 4 , ( 0 , bc_xOrder ) )
                    cls.A[bc_LHS_count,:] = 0
                    cls.B[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:bc_xOrder+1] = numgradient_BC.coeffs
                    cls.E[bc_LHS_count,0] = 1
                    bc_LHS_count += 1

                if i==len(bc_u)-1:
                    numgradient_BC = numericalGradient( 4 , ( bc_xOrder , 0 ) )
                    cls.A[-1-bc_RHS_count,:] = 0
                    cls.B[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,-(bc_xOrder+1):] = numgradient_BC.coeffs
                    cls.E[-1-bc_RHS_count,-1] = 1
                    bc_RHS_count += 1
        if bc_count>4:
            raise ValueError( "Too many boundary conditions present, only 4x are allowed." )
        elif bc_count<4:
            raise ValueError( "Too few boundary conditions present, 4x are required.")
        cls.A.tocsr()
        cls.B.tocsr()
        cls.D.tocsr()
        cls.E.tocsr()
        cls.Ee = cls.E.dot( cls.e )
            
        #
        # Time stepping
        #
        for i in range( cls.Nt-1 ):
            
            #
            # Set up RHS
            #
            cls.v[i,:] = (cls.u[i,:]**2)/2
            cls.Av_k = cls.A.dot( cls.v[i,:] )
            cls.Bu_k = cls.B.dot( cls.u[i,:] )
            cls.f[i,:] = cls.Av_k + cls.Bu_k + cls.Ee

            #
            # Time step
            #
            if i<=n_tOrder or n_tOrder==1:

                cls.phi[i,:] = cls.f[i,:] * cls.dt + cls.u[i,:]
                cls.phi[i,np.abs(cls.phi[i,:])<=zero_tol]=0
                cls.phi[i,-2:]=cls.Ee[-2:]
                cls.phi[i,:2]=cls.Ee[:2]
                cls.u[i+1,:] = spsr.linalg.spsolve( cls.D , cls.phi[i,:] )

            elif n_tOrder==4:
                cls.R_n[i,:] = cls.f[i,:]

                #
                # Perform virtual step (1)
                #
                cls.phi_1[i,:] = (cls.dt/2)*cls.R_n[i,:] + cls.u[i,:]
                #cls.phi_1[i,:] = (cls.dt/2)*spsr.linalg.spsolve( cls.D , cls.R_n[i,:] ) + cls.u[i,:]
                #cls.u_1[i,:] = cls.phi_1[i,:]
                cls.phi_1[i,np.abs(cls.phi_1[i,:])<=zero_tol]=0
                cls.phi_1[i,-2:]=cls.Ee[-2:]
                cls.phi_1[i,:2]=cls.Ee[:2]
                cls.u_1[i,:] = spsr.linalg.spsolve( cls.D , cls.phi_1[i,:] )
                cls.v_1[i,:] = (cls.u_1[i,:] ** 2)/2
                cls.R_1[i,:] = cls.A.dot( cls.v_1[i,:] ) + cls.B.dot( cls.u_1[i,:] ) + cls.Ee

                #
                # Perform virtual step (2)
                #
                cls.phi_2[i,:] = (cls.dt/2)*cls.R_1[i,:] + cls.u[i,:]
                #cls.phi_2[i,:] = (cls.dt/2)*spsr.linalg.spsolve( cls.D , cls.R_1[i,:] ) + cls.u[i,:]
                #cls.u_2[i,:] = cls.phi_2[i,:]
                cls.phi_2[i,np.abs(cls.phi_2[i,:])<=zero_tol]=0
                cls.phi_2[i,-2:]=cls.Ee[-2:]
                cls.phi_2[i,:2]=cls.Ee[:2]
                cls.u_2[i,:] = spsr.linalg.spsolve( cls.D , cls.phi_2[i,:] )
                cls.v_2[i,:] = (cls.u_2[i,:] ** 2)/2
                cls.R_2[i,:] = cls.A.dot( cls.v_2[i,:] ) + cls.B.dot( cls.u_2[i,:] ) + cls.Ee

                #
                # Perform virtual step (3)
                #
                cls.phi_3[i,:] = cls.dt*cls.R_2[i,:] + cls.u[i,:]
                #cls.phi_3[i,:] = cls.dt*spsr.linalg.spsolve( cls.D , cls.R_2[i,:] ) + cls.u[i,:]
                #cls.u_3[i,:] = cls.phi_3[i,:]
                cls.phi_3[i,np.abs(cls.phi_3[i,:])<=zero_tol]=0
                cls.phi_3[i,-2:]=cls.Ee[-2:]
                cls.phi_3[i,:2]=cls.Ee[:2]
                cls.u_3[i,:] = spsr.linalg.spsolve( cls.D , cls.phi_3[i,:] )
                cls.v_3[i,:] = (cls.u_3[i,:] ** 2)/2
                cls.R_3[i,:] = cls.A.dot( cls.v_3[i,:] ) + cls.B.dot( cls.u_3[i,:] ) + cls.Ee

                #
                # Finish the time step
                #
                cls.phi[i,:] = cls.u[i,:] + (cls.dt/6)*( cls.R_n[i,:] + 2*cls.R_1[i,:] + 2*cls.R_2[i,:] + cls.R_3[i,:] )
                cls.phi[i,np.abs(cls.phi[i,:])<=zero_tol]=0
                cls.phi[i,-2:]=cls.Ee[-2:]
                cls.phi[i,:2]=cls.Ee[:2]
                cls.u[i+1,:] = spsr.linalg.spsolve( cls.D , cls.phi[i,:] )

                






                    




        print("Hello there")


