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

                                        "CN"    - Crank-Nicolson, central in time and space.

                                        "DF"    - Dufort-Frankel, central in time and space. With
                                                    a 2/3 time step.

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

        [A]<u> = <b> = [C]<v> + [D]<w>

        Where 

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
        if ( cls.solver == "ftcs" ):
            num_gradient = numericalGradient( 2 , ( 1 , 1 ) )
            num_gradient.formMatrix( cls.Nx )
            cls.grad_matrix_C = num_gradient.gradientMatrix

            cls.time_gradient_C = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        elif cls.solver == "cn":
            num_gradient = numericalGradient( 2 , ( 1 , 1 ) )
            num_gradient.formMatrix( cls.Nx )
            cls.grad_matrix_C = (1/2) * num_gradient.gradientMatrix

            cls.time_gradient_C = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        elif cls.solver == "df":
            cls.grad_matrix_C = spsr.dia_matrix( ( 2 * np.ones( ( 2 , cls.Nx ) ) , [-1,1] ) , shape = ( cls.Nx , cls.Nx ) )
            cls.time_gradient_C = spsr.dia_matrix( ( np.zeros( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        else:
            raise ValueError("Invalid solver selected.")
        cls.C_raw = cls.time_gradient_C + cls.grad_matrix_C * cls.S

        #
        # Calculate the A matrix to be used in the stime stepping
        #
        if cls.solver == "ftcs":
            cls.time_gradient_A = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )
            cls.grad_matrix_A = spsr.dia_matrix( ( np.zeros( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        elif cls.solver == "cn":
            num_gradient = numericalGradient( 2 , ( 1 , 1 ) )
            num_gradient.formMatrix( cls.Nx )
            cls.grad_matrix_A = (-1/2) * num_gradient.gradientMatrix
            cls.time_gradient_A = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        elif cls.solver == "df":
            cls.time_gradient_A = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )
            cls.grad_matrix_A = spsr.dia_matrix( ( 2 * np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        else:
            raise ValueError("Invalid solver selected. How did you make it this far?")
        cls.A_raw = cls.time_gradient_A + cls.grad_matrix_A * cls.S

        #
        # Calculate the D matrix to be used in the time stepping
        #
        if ( cls.solver == "ftcs" ) | ( cls.solver == "cn" ):
            cls.time_gradient_D = spsr.dia_matrix( ( np.zeros( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )
            cls.grad_matrix_D = spsr.dia_matrix( ( np.zeros( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        elif cls.solver == "df":
            cls.time_gradient_D = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )
            cls.grad_matrix_D = spsr.dia_matrix( ( -2 * np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )
        else:
            raise ValueError("Invalid solver selected. How did you make it through 2 checks?")
        cls.D_raw = cls.time_gradient_D + cls.grad_matrix_D * cls.S

        #
        # Include Boundary Conditions
        #
        C_csr = cls.C_raw.tolil()
        C_csr[0,0] = 1
        C_csr[0,1:] = 0
        C_csr[-1,-1] = 1
        C_csr[-1,:-1] = 0
        cls.C = C_csr.todia()
        A_csr = cls.A_raw.tolil()
        A_csr[0,0] = 1
        A_csr[0,1:] = 0
        A_csr[-1,-1] = 1
        A_csr[-1,:-1] = 0
        cls.A = A_csr.todia()
        D_csr = cls.D_raw.tolil()
        D_csr[0,0] = 0
        D_csr[0,1:] = 0
        D_csr[-1,-1] = 0
        D_csr[-1,:-1] = 0
        cls.D = D_csr.todia()

        #
        # Time-Marching Solve
        #
        for i in range( cls.Nt - 1 ):

            # Set up previous time step data
            v = cls.u[i]
            b = cls.C.dot( v )

            # Set up (2x) previous time step data
            if i > 1 :
                w = cls.u[i-1]
                b += cls.D.dot( w )
            elif cls.D.sum() > 0:
                w = cls.u[i]
                b += cls.D.dot( w )

            cls.u[i+1,:] = spsr.linalg.spsolve( cls.A , b )

    def exact( cls , m ):
        """
        This method calculates an exact solution to the heat equation.

        Args:
            m (int):    The number of terms in the exponential Fourier series.

        """

        cls.u_exact = np.zeros( np.shape( cls.u ) )
        cls.u_exact[0,...] = cls.u[0,...]

        cls.L = np.max( cls.x ) - np.min( cls.x )
        T = np.max( cls.t ) - np.min( cls.t )

        #Sigstore = np.zeros( ( cls.Nt , m ,) + np.shape( cls.u_exact[0,...] ) )
        cls.mult = np.zeros( np.shape( cls.u ) )

        for i in range( 1 , cls.Nt ):
            Sigmas = np.zeros( (m,) + np.shape( cls.u_exact[i-1,...] ) )
            ms = 2 * np.arange( m ) + 1
            for ii , mm in enumerate( ms ):
                #print("mm:\t"+str(mm))
                exponent = np.exp( - cls.alpha * cls.t[i] * ( ( mm * np.pi / cls.L ) ** 2 ) )
                #exponent = - cls.alpha * cls.t[i] * ( ( mm * np.pi / L ) ** 2 )
                #exponent = ( ( mm * np.pi / L ) ** 2 )
                #exponent = 1
                #print("\tExponent shape:\t"+str(np.shape(exponent)))
                #print("\tExponent:\t"+str(exponent))
                amplitude = ( 1 - ( -1 ** mm ) ) / ( mm * np.pi )
                #amplitude = 1
                #print("Amplitude shape:\t"+str(np.shape(amplitude)))
                #print("\tAmplitude:\t"+str(amplitude))
                sine = np.sin( mm * np.pi * cls.x / ( cls.L ) )
                #sine = 1
                #print("Sine shape:\t"+str(np.shape(sine)))
                #print("\tSine:\t"+str(sine))
                Sigmas[ii,...] = exponent * amplitude * sine

            cls.mult[i,...] = np.sum( Sigmas , axis = 0 )
            cls.u_exact[i,...] = cls.u_exact[0,0] - 2 * ( cls.u[0,0] - cls.u[0,...] ) * cls.mult[i,...]
            #cls.u_exact[i,...] = np.sum( Sigmas , axis = 0 )
            
            #cls.Sigstore[i,...] = Sigmas

    def error( cls ):
        """
        This method serves as the calculation of error from the exact solution.

        Note:   The method "exact(m)" must be run before this


        """

        cls.error_raw = cls.u_exact - cls.u
        cls.error_abs = np.abs( cls.error_raw )
        cls.error_inf = np.max( cls.error_raw , axis = 0 )
        cls.error_rms = np.linalg.norm( cls.error_raw , axis = 1 )

    def amplificationFactor( cls , error_type = "raw" , timestep = None ):
        """
        This method calculates the amplification factor from the error,

        Note:   The method "error()" must be run before this

        Args:

        error_type (str, optional) :    Which error the method will use to calculate the 
                                            amplification factor. The valid options are:

                                        *"raw": Raw error values

                                        "abs":  Absolute value of error

                                        "inf":  Maximum error value for the time point

                                        "rms":  The root mean square error for the time point

                                        Not case sensitive.

        """

        if error_type.lower()=="raw":
            if timestep:
                error_use = cls.error_raw[timestep]
            else:
                error_use = np.sum( cls.error_raw , axis = 0 ) / len( cls.t )
        elif error_type.lower()=="abs":
            if timestep:
                error_use = cls.error_abs[timestep]
            else:
                error_use = np.sum( cls.error_abs , axis = 0 ) / len( cls.t )
        elif error_type.lower()=="inf":
            error_use = cls.error_inf
        elif error_type.lower()=="rms":
            error_use = cls.error_rms
        else:
            raise ValueError( "Invalid error type selected" )
        
        """
        cls.G = np.zeros( ( len( error_use ) - 1 ) )
        for i in range( len( cls.G ) ):
            cls.G[i] = error_use[i+1] / error_use[i]
        """
        cls.G = np.fft.rfft( error_use ) / ( np.fft.rfft( error_use )[0] )

        A_e =  np.fft.rfft( cls.u_exact ) - np.fft.rfft( cls.u )
        cls.A_error = A_e


##################################################################################################
#
# Burgers Equation Object
#
##################################################################################################

class burgersEquation:
    """
    This object allows a user to solve a Burger's equation. See HW3 for more detail.

    """
    def __init__( self , x , u_0 , t_domain , dt=None , C=None , solver="lax" , nu=0.0 ):
        """
        Initialize the Burger's equation object.

        Args:
            x [float]:  [m] The spatial mesh that will be used for the Burger's equation solve.

                        Note as of 2024/10/31:  Must be uniform mesh.

            u_0 [float]:    [m/s] The function values for the Burger's equation solve. Must
                                correspond to the mesh in "x".

            t_domain (float):   The (2x) entry tuple that describes the time domain that the
                                    solve will be preformed over. The entires must be:

                                ( t_start , t_end )

            dt (float, optional):   [s] The uniform time step. Must be numerical value if "C" is
                                        None. Defaults to None.

            C (float, optional):    [m/s] The C factor of the Burger's equation solve. Must be
                                        numerical value if "dt" is None. Defaults to None.

            solver (str, optional): The solver that will be used to solve the Burger's equation.
                                        The valid options are:

                                    *"LAX": Lax method.
                                        
                                    Defaults to "lax". Not case sensitive.

            nu (float, optional):   [m2/s] The dissipation of the Burger's equation. The default
                                        is 0, which will be an inviscid case.

        """

        if not np.shape( x )==np.shape( u_0 ):
            raise ValueError("x and u_0 must be the same shape.")
        
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
        if C:
            self.C = C
            if dt:
                raise ValueError( "S is present along with dt. Problem is overconstrained. Only one of C and dt must be present." )
            else:
                self.dt = self.C * self.dx
        else:
            self.dt = dt
            self.C = self.dt / self.dx

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
        self.nu = nu
    
    def solve( cls , N_spatialorder=1 , N_timeorder=1 , N_spatialBCorder=None , BC="consistent" ):
        """
        This method solves the Burger's equation for the object according to the inputs 
            to the object and method.

        There are a few things to note with the method. First, the system of equations is
            described as linear equations stored in a diagonal-sparse matrix supplied by SciPy.
            This is done to avoid using extremely large matrices that are stored.

        The system of linear equations can be simply represented as follows:

        [A]<u> = <b> = [C]<v> + [D]<w>

        Here, <v> is the previous time step and <w> is the previous time step squared, in
            accordance tot he flux transfer method.

        This method will march in time

        Args:
            N_spatialorder (int, optional): Spatial order for the solve. Defaults to 1.

            N_timeorder (int, optional):    Time order for the solve. Defaults to 1.

            N_spatialBCorder (int, optional):   The order of the boundary conditions of the 
                                                    spatial gradients. Defaults to None, which 
                                                    makes the boundary conditions gradients the half
                                                    of "N_spatialorder".
        
        """

        #
        # Calculate A matrix
        #
        if cls.solver.lower()=="lax":
            cls.A_matrix = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        #
        # Calculate C matrix
        #
        if cls.solver.lower()=="lax":
            cls.C_matrix = (1/2) * spsr.dia_matrix( ( [ np.ones( cls.Nx ) , np.ones( cls.Nx ) ] , [-1,1] ) , shape = ( cls.Nx , cls.Nx ) )

            if not cls.nu==0:
                cls.visc_grad = numericalGradient( 2 , ( N_spatialorder//2 , N_spatialorder//2 ) )
                cls.visc_grad.formMatrix( cls.Nx )
                cls.C_matrix = cls.C_matrix + cls.nu * cls.visc_grad.gradientMatrix

        #
        # Calculate D matrix
        #
        if cls.solver.lower()=="lax":
            cls.num_grad = numericalGradient( 1 , ( N_spatialorder//2 , N_spatialorder//2 ) )
            cls.num_grad.formMatrix( cls.Nx )
            cls.D_matrix = cls.num_grad.gradientMatrix


            if N_spatialBCorder:
                cls.D_matrix = cls.D_matrix.tolil()

                # The LHS boundary condition
                for i in range( N_spatialorder//2 ):
                    #print("i:{x}".format(x=i))
                    #N_LHS_order = N_spatialorder
                    N_LHS_order = N_spatialBCorder
                    cls.num_grad_LHS = numericalGradient( 1 , ( i , N_LHS_order-i ) )
                    cls.D_matrix[i,:]=0
                    #"""
                    cls.D_matrix[i,i:i+N_LHS_order+1]=cls.num_grad_LHS.coeffs
                    #"""
                    #cls.D_matrix[i,i:(i+N_spatialBCorder+1)]=cls.num_grad_LHS.coeffs

                # The RHS boundary condition 
                for i in range( N_spatialorder//2 ):
                    #N_RHS_order = N_spatialorder
                    N_RHS_order = N_spatialBCorder
                    cls.num_grad_RHS = numericalGradient( 1 , ( N_RHS_order-i , i ) )
                    cls.D_matrix[-i-1,:]=0
                    #"""
                    if i==0:
                        cls.D_matrix[-1,-1-N_RHS_order:]=cls.num_grad_RHS.coeffs
                    else:
                        cls.D_matrix[-i-1,-1-N_RHS_order-i:-i]=cls.num_grad_RHS.coeffs
                    #"""
                    #cls.D_matrix[-i-1,-1-N_RHS_order:]=cls.num_grad_RHS.coeffs
                #"""

                cls.D_matrix = cls.D_matrix.todia()

            cls.D_matrix = -(cls.C) * cls.D_matrix

        #
        # Set up boundary conditions
        #
        if BC.lower()=="consistent":
            cls.C_matrix = cls.C_matrix.tolil()
            cls.D_matrix = cls.D_matrix.tolil()
            cls.C_matrix[0,0]=1
            cls.C_matrix[0,1:]=0
            cls.C_matrix[-1,:]=0
            #cls.C_matrix[-1,-2]=1
            cls.C_matrix[-1,-1]=1
            cls.C_matrix[-1,:] = cls.C_matrix[-1,:].toarray() / np.sum( cls.C_matrix[-1,:].toarray() )
            cls.D_matrix[0,:]=0
            #cls.D_matrix[-1,:]=0
            if not cls.nu==0:
                cls.C_matrix[-1,:]=0
                cls.C_matrix[-1,-1]=1
                cls.D_matrix[-1,:]=0

            cls.C_matrix = cls.C_matrix.todia()
            cls.D_matrix = cls.D_matrix.todia()

        
        #
        # Initialize vectors
        #
        cls.v = np.zeros_like( cls.u )
        cls.w = np.zeros_like( cls.u )
        cls.b = np.zeros_like( cls.u )
        cls.b1 = np.zeros_like( cls.u )
        cls.b2 = np.zeros_like( cls.u )

        #
        # Time stepping
        #
        for i in range( len( cls.t )-1 ):

            # Calculate v vector
            cls.v[i,...] = cls.u[i,...]

            # Calculate w vector
            cls.w[i,...] = ( cls.u[i,...] ** 2 )/2

            # Calculate b vector
            cls.b1[i,...] = cls.C_matrix.dot( cls.v[i,...] ) 
            cls.b2[i,...] = cls.D_matrix.dot( cls.w[i,...] )
            cls.b[i,...] = cls.b1[i,...] + cls.b2[i,...]

            # Solve u = A\b
            cls.u[i+1,:] = spsr.linalg.spsolve( cls.A_matrix , cls.b[i,...] )

        
##################################################################################################
#
# Advection Equation Object
#
##################################################################################################

class advectionEquation:
    """
    This object allows the user to solve the advection equation.

    """
    def __init__( self , x , u_0 , c , t_domain , dt=None , C=None , solver="upwind" , nu=0.0 ):
        """
        Initialize the Advection equation object.

        Args:
            x [float]:  [m] The spatial mesh that will be used for the Advection equation solve.

                        Note as of 2024/10/31:  Must be uniform mesh.

            u_0 [float]:    [?] The function values for the Advection equation solve. Must
                                correspond to the mesh in "x".

            c (float):  [m/s] The velocity of the wave/scalar that is being advected.

            t_domain (float):   The (2x) entry tuple that describes the time domain that the
                                    solve will be preformed over. The entires must be:

                                ( t_start , t_end )

            dt (float, optional):   [s] The uniform time step. Must be numerical value if "C" is
                                        None. Defaults to None.

            C (float, optional):    [m/s] The C factor of the Advection equation solve. Must be
                                        numerical value if "dt" is None. Defaults to None.

            solver (str, optional): The solver that will be used to solve the Advection equation.
                                        The valid options are:

                                    *"upwind": Upwind method.
                                        
                                    Defaults to "upwind". Not case sensitive.

            nu (float, optional):   [m2/s] The dissipation of the Advection equation. The default
                                        is 0, which will be an inviscid case.

        """

        if not np.shape( x )==np.shape( u_0 ):
            raise ValueError("x and u_0 must be the same shape.")
        
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
        if C:
            self.C = C
            if dt:
                raise ValueError( "S is present along with dt. Problem is overconstrained. Only one of C and dt must be present." )
            else:
                self.dt = self.C * self.dx
        else:
            self.dt = dt
            self.C = self.dt / self.dx

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
        self.c = c

        #
        # Set up solver
        #
        self.solver = solver.lower()
        self.nu = nu

    def solve( cls , N_spatialorder=1 , N_timeorder=1 , N_spatialBCorder=None , BC="consistent" ):
        """
        This method solves the Burger's equation for the object according to the inputs 
            to the object and method.

        There are a few things to note with the method. First, the system of equations is
            described as linear equations stored in a diagonal-sparse matrix supplied by SciPy.
            This is done to avoid using extremely large matrices that are stored.

        The system of linear equations can be simply represented as follows:

        [A]<u> = <b> = [C]<v> + [D]<w>

        Here, <v> is the previous time step, and <w> is the time step before that.

        This method will march in time

        Args:
            N_spatialorder (int, optional): Spatial order for the solve. Defaults to 1.

            N_timeorder (int, optional):    Time order for the solve. Defaults to 1.

            N_spatialBCorder (int, optional):   The order of the boundary conditions of the 
                                                    spatial gradients. Defaults to None, which 
                                                    makes the boundary conditions gradients the half
                                                    of "N_spatialorder".
        
        """

        #
        # Calculate A matrix
        #
        if cls.solver.lower()=="upwind":
            cls.A_matrix = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        if cls.solver.lower()=="leapfrog":
            cls.A_matrix = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        #
        # Calculate C matrix
        #
        if cls.solver.lower()=="upwind":
            cls.C_matrix = cls.c * cls.C * spsr.dia_matrix( ( [ np.ones( cls.Nx ) , -np.ones( cls.Nx ) ] , [-1,0] ) , shape = ( cls.Nx , cls.Nx ) ) + spsr.dia_matrix( ( np.ones( cls.Nx ) , 0 ) , shape = ( cls.Nx , cls.Nx ) )

            if not cls.nu==0:
                cls.visc_grad = numericalGradient( 2 , ( N_spatialorder//2 , N_spatialorder//2 ) )
                cls.visc_grad.formMatrix( cls.Nx )
                cls.C_matrix = cls.C_matrix + cls.nu * cls.visc_grad.gradientMatrix

        if cls.solver.lower()=="leapfrog":
            cls.C_matrix = -cls.c * cls.C * spsr.dia_matrix( ( [ -np.ones( cls.Nx ) , np.ones( cls.Nx ) ] , [-1,1] ) , shape = ( cls.Nx , cls.Nx ) )

        #
        # Calculate D matrix
        #
        if cls.solver.lower()=="upwind":
            cls.D_matrix = spsr.dia_matrix( ( np.zeros( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        if cls.solver.lower()=="leapfrog":
            cls.D_matrix = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )


        #
        # Set up boundary conditions
        #
        if BC.lower()=="consistent":
            cls.C_matrix = cls.C_matrix.tolil()
            cls.D_matrix = cls.D_matrix.tolil()

            if cls.solver.lower()=="upwind":
                cls.C_matrix[0,0]=1
                cls.C_matrix[0,1:]=0
            
            if cls.solver.lower()=="leapfrog":
                cls.C_matrix[0,0]=1
                cls.C_matrix[0,1:]=0
                cls.C_matrix[-1,-2]=1/2
                cls.D_matrix[0,:]=0
                cls.D_matrix[-1,-1]=1/2


            cls.C_matrix = cls.C_matrix.todia()
            cls.D_matrix = cls.D_matrix.todia()

        
        #
        # Initialize vectors
        #
        cls.v = np.zeros_like( cls.u )
        cls.w = np.zeros_like( cls.u )
        cls.b = np.zeros_like( cls.u )
        cls.b1 = np.zeros_like( cls.u )
        cls.b2 = np.zeros_like( cls.u )

        #
        # Time stepping
        #
        for i in range( len( cls.t )-1 ):

            # Calculate v vector
            cls.v[i,...] = cls.u[i,...]

            # Calculate w vector
            if i==0:
                cls.w[i,...] = cls.u[i,...]
            else:
                cls.w[i,...] = cls.u[i-1,...]

            # Calculate b vector
            cls.b1[i,...] = cls.C_matrix.dot( cls.v[i,...] ) 
            cls.b2[i,...] = cls.D_matrix.dot( cls.w[i,...] )
            cls.b[i,...] = cls.b1[i,...] + cls.b2[i,...]

            # Solve u = A\b
            cls.u[i+1,:] = spsr.linalg.spsolve( cls.A_matrix , cls.b[i,...] )

