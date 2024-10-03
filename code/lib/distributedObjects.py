"""
DISTRIBUTED OBJECTS

Author:     Matthew Holland

This library contains objects that facilitate the rest of the code to gather and manipulate data.

"""

import numpy as np
import scipy.special as spsp
import scipy.sparse as spsr
from numba import njit, prange, jit

from distributedFunctions import *

#######################################################################################
#
# Numerical Gradient Objects
#
#######################################################################################

class numericalGradient:

    def __init__( self , derivativeOrder , template ):
        """

        This object contains the data pertaining to a numerical gradient

        Args:
            derivativeOrder (int):  The order of the derivative that will be used.

            template ((int)):       The terms in the template that will be used for the
                                        gradient. This will be a tuple of (2x) entries.
                                        The first entry is the number of entries on the 
                                        LHS of the reference point. The second/last 
                                        entry is the number of entries on the RHS of
                                        the reference point.

        Attributes:

            derivativeOrder <-  Args of the same

            template        <-  Args of the same

            coeffs [float]: The coefficients of the numerical gradient according to the
                                template that was put in the object.

        """

        factorial_numba(1)

        if len( template ) > 2:
            raise ValueError( "Too many values in \"template\". Must be 2 entries." )
        elif len( template ) < 2:
            raise ValueError( "Too few values in \"template\". Must be 2 entries." )

        self.derivativeOrder = derivativeOrder
        self.template = template

        self.coeffs = gradientCoefficients( self.derivativeOrder , self.template[0] , self.template[1] , self.derivativeOrder )
        self.coeffs_LHS = gradientCoefficients( self.derivativeOrder , 0 , self.template[0] + self.template[1] , self.derivativeOrder )
        self.coeffs_RHS = gradientCoefficients( self.derivativeOrder , self.template[0] + self.template[1] , 0 , self.derivativeOrder )

    def formMatrix( cls , nPoints , acceleration = None ):
        """

        Form the matrix that calculates the gradient defined by the object. Will follow
            the format:

        [A]<u>=<u^(f)>, where f is the order of the derivative, representing such.

        It will store the [A] is the diagonal sparse format provided by SciPy.sparse

        Args:
            nPoints (int):  The number of points in the full mesh.

            accelerateion (str , optional):    The acceleration method to improve the performance of calculating the
                                        matrix. The valid options are:

                                    - *None :    No acceleration

        Attributes:
            gradientMatrix <Scipy DIA Sparse>[float]:   The matrix to find the gradients.

        
        """

        #
        # Place the data into a CSR matrix
        #
        row = []
        col = []
        data = []
        for j in range( nPoints ):
            #print("j:\t{x}".format(x=j))
            row_array = np.zeros( nPoints )
            if j < cls.template[0]:
                row_array[j:(j+len(cls.coeffs_LHS))] = cls.coeffs_LHS
            elif j >= nPoints - cls.template[0]:
                row_array[(j-len(cls.coeffs_RHS)+1):(j+1)] = cls.coeffs_RHS
            else:
                row_array[(j-cls.template[0]):(j+cls.template[1]+1)] = cls.coeffs
            #print("\trow array:\t"+str(row_array))

            row_cols_array = np.nonzero( row_array )[0]
            row_rows_array = np.asarray( [j] * len( row_cols_array ) , dtype = np.int64 )
            row_data_array = row_array[row_cols_array]
            #print( "\tColumns of non-zero:\t"+str(row_cols_array))
            #print( "\tData of non-zero:\t"+str(row_data_array))

            row += list( row_rows_array )
            col += list( row_cols_array )
            data += list( row_data_array )

        cls_data = np.asarray( data )
        cls_row = np.asarray( row , dtype = np.int64 )
        cls_col = np.asarray( col , dtype = np.int64 )
        #print("\nFinal Data:\t"+str(cls_data))
        #print("Final Rows:\t"+str(cls_row))
        #print("Final Columns:\t"+str(cls_col))

        gradientMatrix_csr = spsr.csr_matrix( ( cls_data , ( cls_row , cls_col ) ) , shape = ( nPoints , nPoints ) )

        #
        # Transfer data to DIA matrix
        #
        cls.gradientMatrix = gradientMatrix_csr.todia()

    def gradientCalc( cls , x , f_x , method = "native" ):
        """

        This method calculates the gradient associated with the discrete values entered into the method.

        Args:
            x [float]:      The discrete values in the domain to calculate the derivative over.

            f_x [float]:    The discrete values in the range to calculate the derivative over.
            
            method (str, optional):     The method of how the gradient will be calculated. The valid options
                                            are:

                                        - *"native" :   A simple matrix multiplication will be used.

                                        - "loop" :  Loop through the rows. Will transfer the matrix to CSR
                                                        to index through rows.

                                        Not case sensitive.

        Returns:
            gradient [float]:   The gradient of the function that was input to the method.

        """

        if len( f_x ) != len( x ):
            raise ValueError( "Lengths of input discrete arrays are not the same." )

        gradient = np.zeros( np.shape( f_x ) )
        dx = np.mean( np.gradient( x ) )
        cls.formMatrix( len( f_x ) )

        if method.lower()=='loop':

            for i , x_i in enumerate( x ):
                #print("i:\t{x}".format(x=i))
                csr_gradient = cls.gradientMatrix.tocsr()
                row = csr_gradient.getrow(i)
                #print("\tRow:\t"+str(row))
                #print("\tf(x):\t"+str(f_x))
                top = row * f_x
                #print("\tTop Portion:\t"+str(top))
                #print("\tdx:\t"+str(dx))
                gradient[i] = top / dx

        elif method.lower()=='native':

            gradient = cls.gradientMatrix.dot( f_x ) / dx

        else:

            raise ValueError( "Invalid method selected" )
        
        return gradient
    
##################################################################################################
#
# Heat Equation Object
#
##################################################################################################

class heatEquation:

    def __init__( self , x , u_0 , t_domain , alpha = None , dt = None , S = None , 
                 solver = "FTCS" , BCs_maintained = True ):
        """

        This object provides a solver for the heat equation. 

        Heat equation

        del(u)/del(t) = alpha * del^2(u)/del(x)^2

        A note on units:    Units must either be in SI or in an equivalent unit system.

        Args:
            x [float]:      [m] The spatial mesh that will be used for the heat equation solve.

                            Note as of 2024/10/03:  Must be uniform mesh.

            u_0 [float]:    [?] The function values for the heat equation solve. Must correspond 
                                to the mesh in "x".

            t_domain (float):   The (2x) entry tuple that describes the time domain that the solve
                                    will be performed over. The entries must be:

                                ( t_start , t_end )

            alpha (float, optional):    [m2/s]The dissipation coefficient value. Must be numerical
                                            value if "S" is None.
                                            
                                        *None.

            dt (float, optional):       [s] The uniform time step. Must be numerical value is "S" 
                                            is None. 
                                            
                                        *None.

            S (float, optional):        The stability factor of the heat equation solve. Must be
                                            numerical value if "alpha" and "dt" are None.
                                             
                                        *None.

            solver (string, optional):  The solver that will be used to solve the heat equation.
                                            The valid options are:

                                        *"FTCS" - Forward in Time, Central in Space.

                                        Not case sensitive.

            BCs_maintained (boolean, optional): Whether the boundary conditions defined in u_0
                                                    will be maintained. 

        Attributes:

            x   <-  Args of the same

            Nx (float):     The number of points along the 1st coordinate of x.

            dx (float):     The uniform mesh size in space.

            S (float):      The stability factor of the explicit method to be used.

            alpha (float):  The dissipation coefficient

            u_0 <-
            
        """

        if not np.shape( x ) == np.shape( u_0 ):
            raise ValueError( "Function values are of not the same shape." )
        
        #
        # Write domain
        #
        self.x = x
        self.Nx = np.shape( x )[0]

        dx_s = np.gradient( self.x )
        ddx_s = np.gradient( dx_s )
        if ( np.sum( ddx_s ) / np.sum( dx_s ) ) > 1e-3 :
            raise ValueError( "x is not uniform enough." )
        else:
            self.dx = np.mean( dx_s )

        #
        # Sort out time stepping & dissipation
        #
        if S:
            self.S = S
            if alpha and dt:
                raise ValueError( "S is present along with both alpha and dt. Problem is overconstrained. Only one of alpha and dt must be present with S." )
            elif alpha:
                self.alpha = alpha
                self.dt = self.S * ( self.dx ** 2 ) / self.alpha
            elif dt:
                self.dt = dt
                self.alpha = self.S * ( self.dx ** 2 ) / self.dt
        else:
            if alpha and dt:
                self.alpha = alpha
                self.dt = dt
                self.S = self.alpha * self.dt / ( self.dx ** 2 )
            else:
                if alpha:
                    raise ValueError( "alpha is present, but not dt or S. Problem is underconstrained." )
                elif dt:
                    raise ValueError( "dt is present, but not alpha or S. Problem is underconstrained." )
                
        #
        # Set up time domain
        #
        self.t = np.arange( t_domain[0] , t_domain[-1] , self.dt )
        self.Nt = len( self.t )

        #
        # Set up the function values
        #
        self.u = np.zeros( ( self.Nt , self.Nx ) )
        self.u[0,...] = u_0

        #
        # Set up solver
        #
        self.solver = solver.lower()
        self.BC_maintain = BCs_maintained

    def solve( cls ):
        """
        This method solves the heat equation. 

        There are a few things to note with the method. First, the system of equations is
            described as linear equations stored in a diagonal-sparse matrix supplied by SciPy.
            This is done to avoid using extremely large matrices that are stored.

        The system of linear equations can be simply represented as follows:

        [A]<u> = <b> = [C]<v>

        Note as of 2024/10/03:  The C-matrix that is used to calculate the next time step is
                                    calculated outside of the time step due to the uniformity
                                    outside of time. This may need to be changed in the future.

        Attributes:

        time_gradient [SciPy sparse DIA - float]:   The sparse matrix that defines the time 
                                                        gradient's contribution to the C matrix.

        grad_matrix [SciPy sparse DIA - float]:     The sparse matrix that defines the spatial
                                                        gradient's contribution to the C matrix.

        C [SciPy sparse DIA - float]:   The C matrix as described above that allows for a RHS SLE
                                            for solving the SLE.

        <u> [float]:    The function values will be modified in this method.

        """

        #
        # Calculate the C matrix to be used in the time stepping
        #
        if cls.solver == "ftcs":
            num_gradient = numericalGradient( 2 , ( 1 , 1 ) )
            num_gradient.formMatrix( cls.Nx )

            cls.time_gradient = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )
        else:
            raise ValueError("Invalid solver selected.")
        cls.grad_matrix = num_gradient.gradientMatrix
        cls.C_raw = cls.time_gradient + cls.grad_matrix * cls.S

        #
        # Include Boundary Conditions
        #
        C_csr = cls.C_raw.tocsr()
        C_csr[0,0] = 1
        C_csr[0,1:] = 0
        C_csr[-1,-1] = 1
        C_csr[-1,:-2] = 0
        cls.C = C_csr.todia()

        for i in range( cls.Nt - 1 ):
            b = cls.C.dot( cls.u[i] )

            cls.u[i+1,:] = b
            if cls.BC_maintain:
                cls.u[i+1,0] = cls.u[i,0]
                cls.u[i+1,-1] = cls.u[i,-1]

    def exact( cls , m ):
        """
        This method calculates an exact solution to the heat equation.

        Args:
            m (int):    The number of terms in the exponential Fourier series.

        """

        cls.u_exact = np.zeros( np.shape( cls.u ) )
        cls.u_exact[0,...] = cls.u[0,...]

        for i in range( 1 , cls.Nt ):
            Sigmas = np.zeros( (m,) + np.shape( cls.u_exact[i-1,...] ) )
            ms = np.arange( m ) + 1
            for ii , mm in enumerate( ms ):
                L = np.max( cls.x ) - np.min( cls.x )
                T = np.max( cls.t ) - np.min( cls.t )
                exponent = np.exp( - cls.alpha * cls.t[i] * ( ( mm * np.pi / L ) ** 2 ) )
                #print("Exponent shape:\t"+str(np.shape(exponent)))
                #amplitude = ( 1 - ( -1 ** mm ) ) / ( mm * np.pi )
                amplitude = 1
                #print("Amplitude shape:\t"+str(np.shape(amplitude)))
                sine = np.sin( mm * np.pi * cls.x / L )
                #print("Sine shape:\t"+str(np.shape(sine)))
                Sigmas[ii,...] = exponent * amplitude * sine
            cls.u_exact[i,...] = cls.u_exact[i-1,0] + 2 * ( cls.u_exact[0,...] - cls.u_exact[i-1,0] ) * np.sum( Sigmas , axis = 0 )

            



