
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import scipy.stats as stats
import statsmodels.api as sm


class SeasonalModel:
    def __init__(self, data, start_date=1, D=365):
        """
        Non-linear function that replicates the seasonality in spot prices.

        Parameters:
        - data: time-series used for fitting the parameters and calculating the model.
        - D: The period (default: 365).
        """
        
        self.data = data
        self.t = np.arange(len(self.data))

        self.D = D
        '''
        The shift parameter H is useful when the prices data does not start at 1 Jan. In the seasonal function if t=1 whe get the theoretical seasonal
        value for the first day of the year, t=1 we get the second and so on. If the prices start at a different time, like day 200/365 then the 
        algorithm would compare day 200 with theoretical seasonal value at day 1. Hence the shift parameters aligns the two.  
        '''
        self.H = start_date.dayofyear if start_date != 1 else start_date
        self.H = 0 if self.H == 1 else self.H

        self.params = None  # Placeholder for fitted parameters

    def getCurve(self, t = None):
        '''
        Returns:
        A series (curve) of the non-linear seasonal function with fitted parameters.
        '''
        if self.params is None:
            raise ValueError("The model has not been fitted yet. Call 'fit()' first.")
        if t is None:
            t = self.t
        return self.calcModel(t, *self.params)

    def calcModel(self, t, a0, a1, a2, a3, a4, a5):
        """
        Model function with parameters and time index.

        Parameters:
        - t: the x-axis input (time).
        - a0...a5: function parameters.

        Returns:
        The value of the function at the point t.
        """
        return (a0 + a1 * t +
                a2 * np.cos(2 * (np.pi / self.D) * (t + self.H - a3)) +
                a4 * np.cos(4 * (np.pi / self.D) * (t + self.H - a5)))

    def fit(self):
        """
        Fit the model to data by optimizing alpha.

        Parameters:
        - colToFit: name of the column, inside the input dataframe, to use to calculate the parameters. 
        
        Returns:
        - params: The parameters fitted for the model.
        - covariance: Covariance matrix of the fit.
        """ 
        initial_guess = [2.5, 0.001, -0.1, 150, -0.05, 100]

        self.params, covariance = curve_fit(self.calcModel, self.t, self.data, p0=initial_guess)
        
        return self


class MeanReversionModel:
    def __init__(self, data, useAnalytical = True):
        """
        Mean Reversion (Ornsten-Uhlenbek) model.

        Parameters:
        - data: time-series used for fitting the parameters and calculating the model.
        - useAnalytical: set 'True' to have sigma and theta analitically calculated, else use 'False' to obtain them trough MLE. 
        """
        self.df = data.copy()
        self.useAnalytical = useAnalytical

        self.theta_hat = None
        self.theta_mle = None
        
        self.sigma_hat = None
        self.sigma_mle = None
        
    
    def calcEstimators(self, colToCalc='LogSeasonalResiduals'):
        '''
        Calculates the explicit analytical estimators of the Mean Reversion stochastic model.

        Parameters:
        - colToCalc: name of the column, inside the input dataframe, to use to calculate the parameters. 
        '''
        self.residualsCol = colToCalc
        n = len(self.df)
        X = self.df[f'{colToCalc}'].copy()

        tmp_sum = np.sum(X[:-1] * X[1:].values)

        if tmp_sum > 0:
            self.theta_hat = -np.log(tmp_sum / np.sum(X[:-1]**2))
        else:
            self.theta_hat = np.nan

        self.sigma_hat = np.sqrt(
            (2 * self.theta_hat) / ((n - 1) * (1 - np.exp(-2 * self.theta_hat))) *
            np.sum((X[1:].values - X[:-1] * np.exp(-self.theta_hat))**2)
        )

        if self.useAnalytical:
            self.sigma = self.sigma_hat
            self.theta = self.theta_hat 

        return self

    def fitEstimators(self,colToCalc='LogSeasonalResiduals'):
        '''
        Function to fit the estimators to input data using Maximum likelihood estimation (MLE).

        Parameters:
        - colToCalc: name of the column, inside the input dataframe, to use to calculate the parameters. 
        '''
        residuals = self.df[f'{colToCalc}'].copy()
        # Define the OU function to evaluate the conditional mean and variance
        def OU(x, t, x0, theta, sigma, log=False):
            Ex = x0 * np.exp(-theta * t)
            Vx = (sigma**2) / (2 * theta) * (1 - np.exp(-2 * theta * t))
            if log:
                return stats.norm.logpdf(x, loc=Ex, scale=np.sqrt(Vx))
            else:
                return stats.norm.pdf(x, loc=Ex, scale=np.sqrt(Vx))

        # Define the negative log-likelihood function
        def OU_lik(params):
            theta, sigma = params
            # Enforce lower bounds to avoid invalid values
            if theta <= 1e-9 or sigma <= 1e-9:
                return np.inf
            n = len(residuals)
            dt = 1  # Assuming time intervals are uniform and equal to 1
            log_likelihood = -np.sum(
                OU(residuals[1:], dt, residuals[:-1], theta, sigma, log=True)
            )
            return log_likelihood

        # Initial parameter estimates, using parameters calculated analitically as initial guesses
        if self.theta_hat and self.sigma_hat:
            initial_guess = [self.theta_hat, self.sigma_hat]
        else:
            print("MeanReversion Model -> fitEsterminators: using initial guesses for theta and sigma = 0.\n\tUse calcEstimators to use analytical guesses.")
            initial_guess = [0, 0]

        # Bounds for the parameters (theta and sigma must be greater than 0)
        bounds = [(1e-9, None), (1e-9, None)]

        # Perform MLE using the L-BFGS-B optimization method
        result = minimize(
            OU_lik,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds
        )

        self.theta_mle, self.sigma_mle = result.x
        
        if not self.useAnalytical:
            self.sigma = self.sigma_mle
            self.theta = self.theta_mle

        return self
    
    def simulationP(self, initialValue, indexRange, simPaths = 1000):
        '''
        Simulates a chosen amount of paths under the probabilistic measure P.

        Parameters:
        - initialValue: the starting price/value of all simulation paths.
        - indexRange: an array of numbers starting from desidered point. i.e. (0,..,n) or (101,...,n) assuming input dataframe with lengh 100.
        - useAnalytical: set to 'True' to use analytical parameters, set to 'False' to use parameters fitted trough MLE.
        - simPaths: number of simulations paths to calculate.  
        
        Returns:
        - dfSimulation: a dataframe with as many columns as the number of simulation paths.
        '''
        if not (self.sigma and self.theta):
            raise ValueError("You must calibrate the parameters (analitycally or trough MLE) before running the simulation.")

        simSize = len(indexRange)
        process = np.zeros(simSize)
        process[0] = initialValue
        dfSimulation = pd.DataFrame()
        dfSimulation.index = indexRange
        
        simulations_list = []
        # ## Simulating the stochastic OU part of the log price
        stdev = np.sqrt(self.sigma**2 / (2 * self.theta) * (1 - np.exp(-2 * self.theta)))
        for sim in range(simPaths):
            
            # for i in range(simSize-1):
            #    i = i+1
            #    process[i] = np.exp(-self.theta) * process[i-1] + stdev * np.random.normal(0, 1)
            
            # dfSimulation[f'Sim-P_{sim+1}'] = process

            for t in range(1, simSize):
                process[t] = np.exp(-self.theta) * process[t-1] + stdev * np.random.normal(0,1)

            simulations_list.append(pd.Series(process.copy(), name=f'Sim-P_{sim+1}'))
        
        dfSimulation = pd.concat(simulations_list, axis=1)

        return dfSimulation

class GasModel:
    def __init__(self, priceCurve, forwardCurve, seasonalModel, stochasticModel):
        """
        Model of Gas prices that assumes these prices are the sum of a non-linear seasonal function and a stochastic Mean Reverting process (Ornstein-Uhlenbeck).

        Parameters:
        - data: time-series used for fitting the parameters and calculating the model. Must have a 'LogPrice' column.
        - seasonalModel: an instance of seasonalModel class, already calibrated.
        - stochasticModel: an instance of the Mean Reverting stochastic model, already calibrated.
        """

        self.df = priceCurve.copy()
        self.fw_df = forwardCurve.copy()

        self.seasonalModel = seasonalModel
        self.params = self.seasonalModel.params
        self.stochasticModel = stochasticModel
        self.sigma = self.stochasticModel.sigma
        self.theta = self.stochasticModel.theta

        self.currentPriceIndex = len(self.df)-1 # is the last day with a spot price i.e. today 
        self.currentPrice = self.df['Close'].iloc[self.currentPriceIndex] # last observed spot price i.e. today
        self.currentResidual = self.df['LogSeasonalResiduals'].iloc[self.currentPriceIndex] # last observed log-residual (logPrice - seasonality)
        #self.currentResidual = self.df['LogPrice'].iloc[self.currentPriceIndex] - self.seasonalModel.calcModel(self.currentPriceIndex, *self.params) # last observed log-residual (logPrice - seasonality)

        self.firstDayFW = self.fw_df.index[0]
        self.fwLen = len(self.fw_df)
        self.forwardDiscretization = np.arange(self.firstDayFW, self.firstDayFW + self.fwLen) # an array of sequential numbers represinting the date in the forward curve
        self.alpha = np.zeros(self.fwLen) # Array of alphas, must be filled with values that make the theoretical forward curve equal to the real one

        self.forwardCurve = None # where the curve is stored, it'll contain the theoretical fw curve before the model is calibrated

    # Function for the integral
    @staticmethod
    def integral(t_i, t_u, t_d, theta):
        """
        Calculates the integral.
        """
        return np.exp(-theta * (t_i - t_u)) - np.exp(-theta * (t_i - t_d))

    # Function for alpha
    @staticmethod
    def alpha_integral(initialDate, forwardDiscretization, alpha, theta):
        """
        Calculates ALPHA.INTEGRAL.
        """
        forwardCurveSize = len(forwardDiscretization)
        alpha_integral_result = np.zeros(forwardCurveSize)
        
        for i in range(forwardCurveSize):
            t_i = forwardDiscretization[i]
            t_u = forwardDiscretization[:i+1]
            t_d = np.append(initialDate, forwardDiscretization[:i])
            Z = alpha[:i+1] * GasModel.integral(t_i, t_u, t_d, theta)
            alpha_integral_result[i] = np.sum(Z)
        
        return alpha_integral_result

    # Function for sigma.x
    @staticmethod
    def sigma_x(initialDate, forwardDiscretization, sigma, theta):
        """
        Calculates sigma.x.
        """
        return np.sqrt(sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * (forwardDiscretization - initialDate))))


    def simulationP(self, paths):
        '''
        Simulate  a certain amount of paths under the measure P.

        Parameters:
        - paths: the amount of simulations to calculate.

        Returns:
        - simulations:  a dataframe with as many columns as the number of simulation paths.
        '''
        
        days = self.fwLen  # Number of days to simulate
        ## Calculating the seasonality path for the simulation
        #### Index for the dates for the seasonal function (so the seasonal calculation in the simulation starts from desired point)
        dateIndex = np.arange(self.currentPriceIndex, self.currentPriceIndex + days)
        #### Calculating seasonality
        simulationsSeasonality = self.seasonalModel.calcModel(dateIndex, *self.params)
        ### Create simulations using analytical estimators
        simulations = self.stochasticModel.simulationP(self.currentResidual,dateIndex,paths)
        ### Sum seasonality to all simulation paths and then exponentiate to convert from log prices to prices
        simulations = np.exp(simulationsSeasonality[:, np.newaxis] + simulations)

        return simulations
    
    def simulationQ(self, paths):
        '''
        A function that returns a dataframe with as many columns as the desidered number of simulations.
        These simulations are under the risk-neutral measure Q, hence all the simulations paths will converge towards the forward curve.
        
        Parameters:
        - paths: the amount of simulations to calculate.
        '''
        # Define inputs
        days = self.fwLen  # Number of days to simulate
        stdev = np.sqrt(self.sigma**2 / (2 * self.theta) * (1 - np.exp(-2 * self.theta)))

        # Initialize the X array
        X = np.full(paths, self.currentResidual) # an array where current residual is repeated N times

        # Initialize an empty array that will store "paths" amount of arrays with size of "days"
        simulations = np.zeros((days, paths))

        # Simulation loop
        for i in range(days):
            '''
            X is an array with as many values as the number of simulations. Every value is the value of the
            simulation at the time i.
            -> With "np.exp(-self.theta) * X" we're taking previous values as a basis for the new ones.
            '''
            X = (np.exp(-self.theta) * X + 
                self.alpha[i] * (1 - np.exp(-self.theta)) + 
                stdev * np.random.normal(0, 1, paths))
            
            # Add deterministic part to the stochastic part to get final simulated spot price
            simulationRow = np.exp(self.seasonalModel.calcModel(self.currentPriceIndex + i + 1, *self.params) + X)
            simulations[i, :] = simulationRow
            
        # Convert the numpy array to a DataFrame after the simulation loop
        simulations_df = pd.DataFrame(
            simulations,
            columns=[f"Sim-Q_{j+1}" for j in range(paths)]
            )
        
        return simulations_df
    
    def calcForwardCurve(self):
        """
        Calculates the forward price using the spot-forward relationship. This function will return the theoretical forward curve if the alpha array is zero (not calibrated model).
        """
        integral_result = self.alpha_integral(self.currentPriceIndex,self.forwardDiscretization, self.alpha, self.theta)
        sigma_x_result = self.sigma_x(self.currentPriceIndex, self.forwardDiscretization, self.sigma, self.theta)
    
        self.forwardCurve = np.exp(
            self.seasonalModel.calcModel(self.forwardDiscretization, *self.params) +
            self.currentResidual * np.exp(-self.theta * (self.forwardDiscretization - self.currentPriceIndex)) +
            integral_result +
            0.5 * sigma_x_result**2
        )
        
        return self
    
    def calibrateForwardCurve(self):
        '''
        This function will find the optimal array of alphas that will make the theoretical forward curve match the observed forward curve. 
        '''
        
        sigma_x_squared = 0.5 * self.sigma_x(self.currentPriceIndex, self.forwardDiscretization, self.sigma, self.theta)**2
        exp_theta_term = np.exp(-self.theta * (self.forwardDiscretization - self.currentPriceIndex))
        seasonal_term = self.seasonalModel.calcModel(self.forwardDiscretization, *self.params)
        Y = np.log(self.fw_df['Close'].values) - sigma_x_squared - seasonal_term - exp_theta_term * self.currentResidual
        
        for i in range(self.fwLen):
            t_i = self.forwardDiscretization[i]
            t_u = self.forwardDiscretization[:i+1]
            t_d = np.append(self.currentPriceIndex, self.forwardDiscretization[:i])

            Z = np.zeros(i + 1)  # Initialize Z array
            Z = self.alpha[:i+1] * self.integral(t_i, t_u, t_d, self.theta)
            Z[i] = self.integral(t_i, t_i, t_d[-1], self.theta)

            if i == 0:
                self.alpha[i] = Y[i] / Z[0]
            else:
                self.alpha[i] = (Y[i] - np.sum(Z[:i])) / Z[i]
        
        self.forwardCurve = self.calcForwardCurve().forwardCurve

        return self

# ---------------------------------------------------------------------------- #
#                                  LOAD MODELS                                 #
# ---------------------------------------------------------------------------- #

class LoadSeasonalModel:
    def __init__(self, data, D=365):
        """
        Non-linear function that replicates the seasonality in load, considering also the correlation with spot prices.

        Parameters:
        - data: time-series used for fitting the parameters and calculating the model, it must be a dataframe with spot prices and load.
        - D: The period (default: 365).
        - H: Shift parameter (default: 0).
        """
        
        self.df = data.copy()
        
        # Indipendent variables
        self.t = np.arange(len(self.df))
        self.price = self.df['Close']
        self.TimePrice = (self.t, self.price)

        self.D = D
        self.H =  self.df['Date'].iloc[0].dayofyear

        self.params = None  # Placeholder for fitted parameters

    def getCurve(self):
        '''
        Returns:
        A series (curve) of the non-linear seasonal function with fitted parameters.
        '''
        if self.params is None:
            raise ValueError("The model has not been fitted yet. Call 'fit()' first.")
        return self.calcModel(self.TimePrice, *self.params)

    def calcModel(self, TimeAndSpot, a0, a1, a2, a3, a4, beta):
        """
        Model function with parameters and time index.

        Parameters:
        - TimeAndSpot: a touple that contains a time index and a spot price (indipendent variables).
        - a0...a4, beta: function parameters.

        Returns:
        The value of the function at the time index inside the touple.
        """
        t, spot_price = TimeAndSpot
        return (a0 
            + a1 * np.cos((t + self.H - a2) * 2 * np.pi/self.D) 
            + a3 * np.cos((t + self.H - a4) * 4 * np.pi/self.D)
            + beta * np.log(spot_price)) 
            
    def fit(self, colToFit, initial_guess):
        """
        Fit the model to data by optimizing alpha.

        Parameters:
        - colToFit: name of the column, inside the input dataframe, to use to calculate the parameters. 
        - initial_guess: initial guesses for the parameters.
        
        Returns:
        - params: The parameters fitted for the model.
        - covariance: Covariance matrix of the fit.
        """ 
        # The column of the dataframe to be used in order to fit the data
        df_fit = self.df[f'{colToFit}'].dropna().reset_index(drop=True)
        
        self.params, covariance = curve_fit(self.calcModel, self.TimePrice, df_fit, p0=initial_guess)

        return self
    

class LoadSeasonalVolatilityModel:
    def __init__(self, data, D=365):
        """
        Non-linear function that replicates the seasonality volatility in load prices, considering also the correlation with spot prices.

        Parameters:
        - data: time-series used for fitting the parameters and calculating the model, it must be a dataframe with spot prices, load and squared residuals.
        - D: The period (default: 365).
        """
        self.df = data.copy()

        self.D = D
        self.H =  self.df['Date'].iloc[0].dayofyear-1

        self.expectedMonthlyVariance = None 

        self.params = None

    def calcMonthlyVariance(self):        
        date_sq_residuals = self.df['Date'].iloc[1:].reset_index(drop=True)
        sq_residuals_trimmed = self.df['Sq_Residuals2'].iloc[:-1].reset_index(drop=True)

        # Ensure that `Date` is of datetime type.
        # Extract the month from the dates
        months = date_sq_residuals.dt.month
        # Group by month and compute mean
        monthly_mean = sq_residuals_trimmed.groupby(months).mean()

        # Convert to a DataFrame with a descriptive column name
        self.expectedMonthlyVariance = monthly_mean.to_frame(name='Variance')
        #Optionally, set a name for the index (i.e., "Month")
        self.expectedMonthlyVariance.index = np.arange(1,13)
        self.expectedMonthlyVariance.index.name = 'Month'
        return self
    
    def calcModel(self, t, a0, a1, a2, a3, a4):
            """
            Model function with parameters and time index.

            Parameters:
            - TimeAndSpot: a touple that contains a time index and a spot price (indipendent variables).
            - a0...a4, beta: function parameters.

            Returns:
            The value of the function at the time index inside the touple.
            """
            return (a0 
            + a1 * np.cos((t - a2 - self.H) * 2 * np.pi / self.D) 
            + a3 * np.cos((t - a4 - self.H) * 4 * np.pi / self.D))
    
    def sigma_sq_SeasonalFunction(self,t,params):
        '''
        Seasonal function sigma^2(t) for the load volatility.
        '''
        c0 = params[0]
        c1 = params[1]
        c2 = params[2]
        return c0 + c1 * np.cos((2 * np.pi / 365) * (t - c2))

    def fitSeasonalVolatility(self, initial_guess):
        # ## Creating a vector with mid-month day numbers
        # ## (e.g., 45 = 14 February)
        # We will use values to indicise montly variances
        days = np.array([16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350])
        self.expectedMonthlyVariance['days'] = days
        # Step 3: Fit the model using curve_fit
        self.params, covariance = curve_fit(self.calcModel, self.expectedMonthlyVariance['days'].values, self.expectedMonthlyVariance['Variance'].values, p0=initial_guess)

        return self

    def getCurve(self):
        '''
        Returns:
        A series (curve) of the non-linear load seasonal volatility function with fitted parameters.
        '''
        if self.params is None:
            raise ValueError("The model has not been fitted yet. Call 'fit()' first.")
        
        #curve = np.zeros(np.arange(366))
        curve = self.sigma_sq_SeasonalFunction(np.arange(1,366),self.params)
        
        return curve

    #     # Indipendent variables
    #     self.t = np.arange(len(self.df))
    #     self.price = self.df['Close']
    #     self.TimePrice = (self.t, self.price)

    #     self.D = D
    #     self.H =  self.df['Date'].iloc[0].dayofyear

    #     self.params = None  # Placeholder for fitted parameters

            
    # def fit(self, colToFit, initial_guess):
    #     """
    #     Fit the model to data by optimizing alpha.

    #     Parameters:
    #     - colToFit: name of the column, inside the input dataframe, to use to calculate the parameters. 
    #     - initial_guess: initial guesses for the parameters.
        
    #     Returns:
    #     - params: The parameters fitted for the model.
    #     - covariance: Covariance matrix of the fit.
    #     """ 
    #     # The column of the dataframe to be used in order to fit the data
    #     df_fit = self.df[f'{colToFit}'].dropna().reset_index(drop=True)
        
    #     self.params, covariance = curve_fit(self.calcModel, self.TimePrice, df_fit, p0=initial_guess)

    #     return self


class LoadStochasticModel:
    def __init__(self, data, loadSeasonalModel):
        """
        Model for the stochastic component in the load, model is fitted trough AR(1) regression.

        Parameters:
        - data: time-series used for fitting the parameters and calculating the model. Must contain spot prices and load data.
        - LoadSeasonalModel: an instance of LoadSeasonalModel class, already calibrated.
        """

        self.df = data.copy()

        self.seasonalModel = loadSeasonalModel
        self.params = self.seasonalModel.params
        
        self.currentPriceIndex = len(self.df)-1 # is the last day with a spot price i.e. today 
        self.currentPrice = self.df['Close'].iloc[self.currentPriceIndex] # last observed spot price i.e. today

        self.residuals = self.df['Value'] - self.seasonalModel.getCurve() # Delta between consumes and fitted seasonal consumes
        self.residuals2 = None # Delta between residuals and the stochastic component of consumes

        self.AR_coeff = None

    def estimateAR(self):
        '''
        Estimates an AR(1) process for the load.
        '''
        # Create a DataFrame similar to R's y.data
        regression_df = pd.DataFrame() 
        regression_df['TimeIndex'] = np.arange(1,len(self.df))
        regression_df['ShiftedResiduals'] = self.residuals[1:].reset_index(drop=True)
        regression_df['LaggedResiduals'] = self.residuals[:-1].reset_index(drop=True)    
        
        # Fit AR(1) model: y = a0 + a1*y_lag
        X = sm.add_constant(regression_df['LaggedResiduals'])  # Add an intercept term
        model = sm.OLS(regression_df['ShiftedResiduals'], X).fit()

        # Summary of the model
        #print(model.summary())
        # Confidence intervals
        #print(model.conf_int())
        # Coefficients
        self.AR_coeff = model.params
        #print("AR Coefficients:", AR_coeff)

        self.residuals2 = model.resid


        return self

class LoadModel:
    def __init__(self,data,loadSeasonalModel, loadStochasticModel, loadSeasonalVolatilityModel):
        '''
        A model for the load. 

        Parameters:
        - data: a dataframe that contains spot prices and load data.
        - seasonalParams: the paramters extrapolated trough the fitting of the seasonal (detemrinistic) function.
        - loadSeasonalModel: a seasonal (deterministic) model for load.
        - loadStochasticModel: a stochastic model for load fitted trough AR(1).
        - loadSeasonalVolatilityModel: a seasonal (deterministic) model of the volatility in load.
        '''
        
        self.df = data.copy()

        self.seasonalParams = loadSeasonalModel.params
        self.stochasticGamma = loadStochasticModel.AR_coeff.iloc[1]

        self.seasonalVolatilityModel = loadSeasonalVolatilityModel
        self.seasonalVolatilityParams =  loadSeasonalVolatilityModel.params
        

        return
    
    def simulateStochasticPart(self):
        '''
        Simulation of the stochastic part of the load.
        '''

    def betaFunc(self, t , params ,S):
        '''
        Calculates the load model, assumed to be h(t) + beta * ln(SpotPrice)

        Parameters:
        - t: a time index. 
        - params: parameters for the function.
        - S: spot price at given time t.
        '''
        b0, b1, b2, b3, b4, beta = params
        h_beta = (
            b0 +
            b1 * np.cos(2 * np.pi / 365 * (t - b2)) +
            b3 * np.cos(4 * np.pi / 365 * (t - b4)) +
            beta * np.log(S)
        )
        return h_beta

    def simulateLoad(self,simulatedPrice):
        '''
        Given a dataframe with N columns (amount of simulations) and X rows (amount of days simulated), this function will
        return a dataframe with N columns and X rows of simulated load.

        Parameters:
        - N: the number of simulations.
        - simulatedPrice: a dataframe with "N" columns.
        '''
        N = simulatedPrice.shape[1]
        days = len(simulatedPrice)
        # Initialize arrays
        starting_index = simulatedPrice.index[0]
        #Load_sim = np.zeros((N, days))   # To store load simulations
        y = np.zeros(N)                  # Initialize y for stochastic AR(1)
        
        ### Calculating the standard deviations for each day
        sigma_AR = np.sqrt(self.seasonalVolatilityModel.sigma_sq_SeasonalFunction(simulatedPrice.index+1,self.seasonalVolatilityParams))
        simulations = np.zeros((days, N))

        for i in range(days):
            # Simulating the stochastic AR(1) part of the load            
            y = self.stochasticGamma * y + sigma_AR[i] * np.random.normal(0, 1, N)
            # Setting the index on par with simulations index (before it was just from 0 to days because sigma_AR is just
            # an array with lenght = 'days' but not indexed)
            
            # Add the seasonal part to the AR(1) part to get the load
            simulationRow = self.betaFunc(simulatedPrice.index[i]+1, self.seasonalParams,simulatedPrice.iloc[i, :].values) + y
            
            # Store the row in the pre-allocated NumPy array
            simulations[i, :] = simulationRow
            
        # Convert the NumPy array to a DataFrame after the loop
        simulations_df = pd.DataFrame(
            simulations,
            columns=[f"Sim-Load_{j+1}" for j in range(N)]
        )
            
        return simulations_df

    def getLoadSimulation(self):
        '''
        A function that simulates the load curve.
        '''
        simulated_load = pd.DataFrame()
        ### Calculating the standard deviations for each day
        simulated_load['SigmaAR'] = np.sqrt(self.seasonalVolatilityModel.sigma_sq_SeasonalFunction(np.arange(1,len(self.df)+1),self.seasonalVolatilityParams))    

        N = len(self.df.index)
        y = np.zeros(N, dtype=float)
        y[0] = self.df['Value'].iloc[0] - self.betaFunc(self.df.index[0]+1, self.seasonalParams, self.df['Close'].iloc[0])
        
        for i in range(1, N):
            y[i] = self.stochasticGamma * y[i - 1] + simulated_load['SigmaAR'].iloc[i] * np.random.normal(0, 1)

        # ## Adding the seasonal / deterministic part to the stochastic part to get the final simulated loads
        simulated_load['Simulation'] = self.betaFunc(self.df.index+1,self.seasonalParams,self.df['Close']) + y
        #Load.sim <- h.beta.func(Time, h.beta.coeff, Price) + y
        
        return simulated_load[['Simulation']]