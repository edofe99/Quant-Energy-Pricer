import numpy as np
import pandas as pd
from model.models import *
from scipy.optimize import root_scalar
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
# Switch to Tk backend to plot charts
# import matplotlib
# matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt


class Contract:
    def __init__(self, price_data, forward_data, load_data, simulations, risk_free, 
                 contract_start, contract_end,
                 contract_price, hurdle_rate, figures):
        
        self.number_simulations = simulations
        self.risk_free = risk_free
        self.contractStart = contract_start 
        self.contractEnd = contract_end      
        self.contractPrice = contract_price
        self.hurdleRate = hurdle_rate
        self.figures = figures # An array containing the plots

        # ------------------------------ Preparing data ------------------------------ #
        self.spot_df = price_data
        self.spot_df = self.spot_df.rename(columns={self.spot_df.columns[1]: 'Close'})
        
        self.load_df_full = load_data[-1]
        self.load_df = load_data[0]
        self.load_df = self.load_df.rename(columns={self.load_df.columns[1]: 'Value'})

        self.fw_df = forward_data 
        self.fw_df = self.fw_df.rename(columns={self.fw_df.columns[1]: 'Close'})
        
        # Dataframes with all the simulations
        self.simulatedPrice = None
        self.simulatedLoad = None
        
        # -------------------------- Contract Discretization ------------------------- #

        valuationDate = self.spot_df['Date'].iloc[-1] # Data must be updated to today
        startDate = pd.to_datetime(self.contractStart)
        endDate = pd.to_datetime(self.contractEnd)
        startToEnd = (endDate-startDate).days # end date not included
        todayToStart = (startDate-valuationDate).days # end date not included
        
        valuationIndex = self.spot_df.index[-1] # today
        self.startIndex = valuationIndex + todayToStart
        self.endIndex = self.startIndex + startToEnd +1 #WITH +1 WE CONSIDER ALSO THE LAST DAY
        
        self.contractIndex = np.arange(self.startIndex,self.endIndex) # The time inex of the contract 
        self.timeIndex = np.arange(valuationIndex+1,self.endIndex)  # The time index from tomorrow to the end of the contract

        self.discount = np.exp(-self.risk_free * (self.contractIndex-valuationIndex) )# Discount factors aloing contract life

        # ----------------- Simulations across the contract lifespan ----------------- #
        ##### Forward, Spot and Load curves (simulations) along contract life
        self.contractForward = None
        self.contractSpot = None
        self.contractLoad = None

        # Average contract load alogn contract life (averaging all simulations by row)
        self.contractAverageLoad = None

        self.optimalPrice = None # The final contract price

    # ---------------------------------------------------------------------------- #
    #                               Useful functions                               #
    # ---------------------------------------------------------------------------- #

    @staticmethod
    def removeOutliers(data: pd.Series) -> pd.Series:
        '''
        A function that removes outliers from a dataframe column / series using the interquantile methods. 
        Outliers are not removed but replaced with the average of two closet non-outlier observations.

        Parameters:
        - data: a pandas series.

        Returns:
        - data: the same column without outliers.
        '''
        # Calculate the daily logarithmic changes
        #df = data.copy()
        #df = df.dropna().reset_index(drop=True)

        # Calculate the first and third quartiles of the 'LogReturns' column
        first_quartile = data.quantile(0.25)
        third_quartile = data.quantile(0.75)
        interquartile_range = third_quartile - first_quartile

        # Define the lower and upper bounds
        lower_bound = first_quartile - 3 * interquartile_range
        upper_bound = third_quartile + 3 * interquartile_range

        # Function to replace outliers with the average of two closest non-outlier values
        # Function to replace outliers with the average of two closest non-outlier values
        def replace_outliers_with_nearest_avg(series, lower, upper):
            series = series.copy()  # Avoid modifying the original Series
            for i in range(len(series)):
                if series.iloc[i] < lower or series.iloc[i] > upper:
                    # Find the two closest non-outlier neighbors
                    non_outliers = series[(series >= lower) & (series <= upper)]
                    closest_below = non_outliers[non_outliers.index < series.index[i]].iloc[-1] if not non_outliers[non_outliers.index < series.index[i]].empty else series.iloc[i]
                    closest_above = non_outliers[non_outliers.index > series.index[i]].iloc[0] if not non_outliers[non_outliers.index > series.index[i]].empty else series.iloc[i]
                    
                    # Replace the outlier with the average of closest non-outlier neighbors
                    series.iat[i] = (closest_below + closest_above) / 2
            return series

        # Apply the function to 'LogReturns' column
        data = replace_outliers_with_nearest_avg(data, lower_bound, upper_bound)

        #df.to_csv("/home/edoardo/Desktop/Stage Ferretto Edoardo/Codes/Data/spot_no_outliers.csv",index=False)

        return data

    @staticmethod
    def calculate_es(profit, confidence):
        """
        Calculate the Expected Shortfall.

        Parameters:
        - Profit (ndarray): A NumPy array of profit values.
        - confidence (float): The confidence level (e.g., 0.95 for 95%).

        Returns:
        - float: The ES value.
        """
        # Calculate the quantile threshold
        quantile_alpha_pct = np.quantile(profit, 1 - confidence)  # In R: quantile(Profit, confidence)
        # Identify the tail observations (losses below the quantile)
        tail_obs = profit[profit < quantile_alpha_pct]
        # Calculate the mean of the tail observations
        cfetl = np.mean(tail_obs) if len(tail_obs) > 0 else 0  # Handle empty tails gracefully
        
        return cfetl
    
    @staticmethod
    def calculate_var(array, confidence):
        percentile = (1 - confidence) * 100
        var = np.percentile(array, percentile)

        return var
    

    @staticmethod
    def plot_histogram(array,title,xlabel,ylabel,v_line=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(array, bins=100, edgecolor='black', alpha=0.75)
        ax.set_title(title, wrap = True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # If line_x is provided, draw a vertical red line at that position
        if v_line is not None:
            ax.axvline(x=v_line, color='red', linestyle='-', linewidth=2)
        
        return fig
    
    # ---------------------------------------------------------------------------- #
    #                             Simulation functions                             #
    # ---------------------------------------------------------------------------- #
    
    def simulatePrice(self):
        # ----------------------------- Preliminar steps ----------------------------- #

        self.spot_df['Date'] = pd.to_datetime(self.spot_df['Date'])
        self.spot_df['LogPrice'] = np.log(self.spot_df['Close'])
        self.spot_df['LogReturns'] = np.log(self.spot_df['Close'] / self.spot_df['Close'].shift(1))

        # ---------------------------- 1. remove outliers ---------------------------- #
        # Remnove outliers from log returns
        self.spot_df['LogReturns'] = self.removeOutliers(self.spot_df['LogReturns'])
        
        # Calculate again log prices from cleaned log returns
        self.spot_df['LogPrice'] = self.spot_df['LogReturns'] + self.spot_df['LogPrice'].shift(1)
        self.spot_df.loc[0,'LogPrice'] = np.log(self.spot_df['Close'].iloc[0])
        
        # --------------------------- 2. Fit seasonal model -------------------------- #
        start_date = self.spot_df['Date'].iloc[0]
        seasonalModel = SeasonalModel(self.spot_df['LogPrice'].values, start_date, D=365)
        self.spot_df['Seasonal Curve'] = seasonalModel.fit().getCurve()

        # ------------------------- Seasonal Plot Projection ------------------------- #
        start_date = self.spot_df['Date'].iloc[-1]
        # Create date range for 2 years
        dates = pd.date_range(start=start_date, periods=365*3, freq='D')
        self.df_projection = pd.DataFrame({'Date': dates})
        self.df_projection['Seasonal Projection'] = seasonalModel.getCurve(self.df_projection.index)
        
        # ------------------------------- Get residuals ------------------------------ #

        self.spot_df['LogSeasonalResiduals'] =  self.spot_df['LogPrice'] - self.spot_df['Seasonal Curve']
        
        # ----------------------- 3. Model the Stochastic Term ----------------------- #

        # Create and calibrate the mean reversion model, both alitycally and with MLE
        meanReversionModel = MeanReversionModel(self.spot_df).calcEstimators().fitEstimators()
        
        print("Ornestein-Uhlenbek parameters")
        print('\t','Analytical','\t','MLE')
        print(f'Theta\t{meanReversionModel.theta_hat}\t{meanReversionModel.theta_mle}')
        print(f'Sigma\t{meanReversionModel.sigma_hat}\t{meanReversionModel.sigma_mle}')
        
        # Get residuals
        residualsFitted = np.exp(-meanReversionModel.theta) * self.spot_df['LogSeasonalResiduals'].iloc[:-1]    
        self.spot_df['LogResiduals'] = self.spot_df['LogSeasonalResiduals'].iloc[1:].reset_index(drop=True) - residualsFitted
        
        # ------------------------ Stochastic plot projection ------------------------ #

        last_residual = self.spot_df['LogSeasonalResiduals'].iloc[-1]
        self.st_projection = meanReversionModel.simulationP(last_residual,range(len(self.fw_df['Date'])), simPaths=self.number_simulations)
        self.st_projection.index = self.fw_df['Date']
        
        # --------------------------- 4. Simulation under P -------------------------- #

        ## Create the first istance of Gas model
        gasModel = GasModel(self.spot_df,self.fw_df,seasonalModel,meanReversionModel)
        
        self.sample_simulations = gasModel.simulationP(self.number_simulations)
        
        self.expected_price_p = pd.DataFrame(self.sample_simulations.mean(axis=1).values)
        self.expected_price_p.index = self.fw_df['Date']

        # ---------------------- 5. Estimating the drift under Q --------------------- #

        # Calculate theoretical forward price
        self.fw_df['Theoretical curve'] = gasModel.calcForwardCurve().forwardCurve
        # Calibrate the forward curve
        gasModel.calibrateForwardCurve()
        # Calculate the calibrated forward curve 
        self.fw_df['Calibrated curve'] = gasModel.calcForwardCurve().forwardCurve
        self.contractForward = self.fw_df.loc[self.contractIndex, 'Calibrated curve'].reset_index(drop=True)
        # Store alphas
        self.alphas = pd.Series(gasModel.alpha, index = self.fw_df['Date'])
        ## Calculate simulations
        self.simulatedPrice = gasModel.simulationQ(self.number_simulations)
        return self
        

    def simulateLoad(self):

        self.load_df["Date"] = pd.to_datetime(self.load_df["Date"])#, format="%d/%m/%Y")
        self.load_df_full["Date"] = pd.to_datetime(self.load_df_full["Date"])#, format="%d/%m/%Y")
        self.load_df['Value'] = self.removeOutliers(self.load_df['Value'])
        self.load_df_full['Value'] = self.removeOutliers(self.load_df_full['Value'])
        

        df = pd.merge(self.spot_df, self.load_df, on="Date")

        # ------------------------ 1. Fit seasonal load model ------------------------ #
        initial_guess = [ 25, 22, 700, -2, 130, 3 ]

        loadSeasonalModel = LoadSeasonalModel(df) 
        df['Seasonal Curve'] = loadSeasonalModel.fit('Value',initial_guess).getCurve()
        
        # ---------------------- 2. Modeling the stochastic term --------------------- #
        ## Calculate residuals
        loadStochasticModel = LoadStochasticModel(df,loadSeasonalModel)
        df['Residuals'] = loadStochasticModel.residuals
        
        # Estimate the stochastic therm trough AR
        loadStochasticModel.estimateAR()

        df['Residuals2'] = loadStochasticModel.residuals2
        
        df['Sq_Residuals2'] = df['Residuals2']**2

        # -------------------- 3. Modeling the seasonal volatility ------------------- #

        loadSeasonalVolatilityModel = LoadSeasonalVolatilityModel(df)
        expectedSigma = loadSeasonalVolatilityModel.calcMonthlyVariance().expectedMonthlyVariance

        initial_guess  = [0, 2, 50, 1, 40]
        loadSeasonalVolatilityModel.fit(initial_guess)

        self.volatility_df = pd.DataFrame()
        self.volatility_df['FittedVariance'] = loadSeasonalVolatilityModel.getCurve(np.arange(1,366))
        self.volatility_df['MonthlyVariance'] = expectedSigma.set_index('days')['Variance']
        
        # df['Sigma_AR'] = np.sqrt(loadSeasonalVolatilityModel.sigma_sq_SeasonalFunction(np.arange(2,len(df)+2),loadSeasonalVolatilityModel.params))
        df['Sigma_AR'] = np.sqrt(loadSeasonalVolatilityModel.getCurve(np.arange(2,len(df)+2)))
        df['Residuals3'] = df['Residuals2'] / df['Sigma_AR']
        
        # ---------------------- 4. Simulating the customer Load --------------------- #
        
        # Simulating the stochastic AR (1) part of the load
        loadModel = LoadModel(df,loadSeasonalModel,loadStochasticModel,loadSeasonalVolatilityModel)
        simulations = loadModel.simulateLoad(self.simulatedPrice)

        self.simulatedLoad = simulations
        self.simulatedLoad_all = df # Keeping this for a plot

        return self

    # ----------------------- Calculate the contract Price ----------------------- #

    def contractCalculation(self):
        
        # --------------------- 1. Simulating Gas and Load paths --------------------- #
        simulatedPrice = self.simulatePrice().simulatedPrice
        
        # need to keep this for the plot with average simulated price path vs forward
        self.simulatedPrice_all = self.simulatedPrice 

        simulatedPrice = simulatedPrice.iloc[:len(self.timeIndex)] #Cutting out simulations after the end of the contract
        simulatedPrice.index = self.timeIndex # Set index of simulations
        self.simulatedPrice = simulatedPrice
    
        self.simulatedLoad = self.simulateLoad().simulatedLoad
        self.simulatedLoad.index = self.timeIndex

        self.contractSpot =  simulatedPrice.loc[self.startIndex:]
        self.contractLoad =  self.simulatedLoad.loc[self.startIndex:]
        
        averageLoadCurve = self.simulatedLoad.mean(axis=1)
        self.contractAverageLoad = averageLoadCurve.loc[self.startIndex:]

        # ------------------------ 2. Calculate contract price ----------------------- #
        
        fw_df = self.fw_df

        price_df = pd.DataFrame()
        price_df.index = self.simulatedPrice.index
        price_df['ForwardPrice'] = fw_df['Close'].iloc[:len(self.timeIndex)] # Getting forward prices until end of contract
        
        price_df['Expected Load'] = averageLoadCurve.values
        # Cutting out values from tomorrow to the start of the contract: we need only values during contract
        price_df = price_df.loc[self.contractIndex]
        
        P = np.dot(price_df['Expected Load'],price_df['ForwardPrice']) / np.sum(price_df['Expected Load'])
        
        # Function to calculate the profit
        def calculateProfit(contractPrice, nominal=True):
            Profit = np.zeros(self.number_simulations)
            for i in range(len(self.contractIndex)):
                z = self.startIndex+i # the index of the contract
                Profit +=  self.discount[i] * (
                    (contractPrice - self.simulatedPrice.loc[z].values)  
                    * self.simulatedLoad.loc[z].values)
            if nominal:
                return Profit/self.simulatedLoad.sum().values
            else:
                return Profit


        # Function to calculate the RAROC
        def calculateRAROC(contractPrice,confidence):
            """
            Calculate the Risk-Adjusted Return on Capital (RAROC).
            """
            # Calculate profits
            Pi = calculateProfit(contractPrice)
            # Mean profit
            Pi_mean = np.mean(Pi)
            # Conditional expected tail loss
            Psi = self.calculate_es(Pi, confidence)
            # Calculate RAROC
            RAROC = Pi_mean / (Pi_mean - Psi) if (Pi_mean - Psi) != 0 else float('inf')
            return RAROC

        confidence = 0.95
        
        nominal = True # Plot profit in nominal value or in face value
        unit = f'{"€/MWh" if nominal else "€"}'

        # -------------------- Finding the optimal contract price -------------------- #
        
        # Define the function whose root we are solving for
        def objective(C):
            return calculateRAROC(C, confidence) - self.hurdleRate

        # Use root_scalar to find the root within the interval [25, 35]
        result = root_scalar(objective, bracket=[0, 150], method='brentq')

        if not result.converged:
            raise ValueError("Root-finding did not converge.")

        self.optimalPrice = result.root

        #print(f'Best contract price: {optimalPrice}€')

        profit = calculateProfit(self.optimalPrice,nominal)
        expected_profit = np.mean(profit) if nominal else int(np.mean(profit)) 
        raroc = calculateRAROC(self.optimalPrice,confidence)
        self.var = self.calculate_var(profit,confidence)

        # title_chart = f'Profit distribution of a contract with price {self.optimalPrice:.2f}€\
        #         \nExpected profit: {expected_profit:.2f}{unit} | RAROC: {raroc:.2f}\
        #         $VAR(\\alpha={confidence*100:.0f}\\%)={self.var:.2f}{unit}$'
        self.calc_volume_risk()
        
        title_chart = f'Profit distribution with no hedging\n' \
              f'Contract Price: {self.optimalPrice:.2f}€ | ' \
              f'Final Contract Price: {self.optimalPrice+self.volume_risk_var:.2f}€\n' \
              f'Expected profit: {expected_profit:.2f}{unit} | RAROC: {raroc:.2f} | ' \
              f'VAR(α={confidence*100:.0f}%): {self.var:.2f}{unit}'


        fig = self.plot_histogram(profit,title_chart,f'Profit [{unit}]', 'Frequency',self.var)
        self.figures.append(fig)
        
        self.generatePlots()


        return self

    # ------------------------------ Other fonctions ----------------------------- #

    def calc_volume_risk(self):
        # spot_simulations = self.contractSpot.reset_index(drop=True)
        # load_simulations = self.contractLoad.reset_index(drop=True)
        # average_load_curve = self.contractAverageLoad.reset_index(drop=True)        

        # # Step 1: Subtract avg_load from load (broadcasts correctly row-wise)
        # load_deviation = load_simulations.sub(average_load_curve, axis=0)
        # # Step 2: Multiply element-wise with price
        # volume_risk_df = pd.DataFrame(load_deviation.values * spot_simulations.values,
        #                         columns=load_simulations.columns, index=load_simulations.index)
        # # Step 3: Aggregate risk per time step by averaging across simulations
        # self.volumeRisk = volume_risk_df.mean(axis=1).sum()  # Shape: (N,)

        # # pi1 =  spot_simulations.mul(average_load_curve, axis=0)
        # # pi1 = pi1.sum(axis=0)
        # # pi1 = pi1.mean()
        # # # print(pi1)

        # # pi2 = spot_simulations.values * load_simulations.values
        # # pi2 = pi2.sum(axis=0)
        # # print(len(pi2))
        # # pi2 = pi2.mean()
        # # # print(pi2)

        # # print(f'Volume risk: {pi2 - pi1:.2f}')

        spot_simulations = self.contractSpot.reset_index(drop=True)
        load_simulations = self.contractLoad.reset_index(drop=True)
        average_load_curve = self.contractAverageLoad.reset_index(drop=True)

        pi1 =  spot_simulations.mul(average_load_curve, axis=0)
        pi1 = pi1.sum(axis=0).values

        pi2 = spot_simulations.values * load_simulations.values
        pi2 = pi2.sum(axis=0)
        # Sum of all simulations
        # load_sum = load_simulations.sum(axis=0).values
        # pi2 = pi2 #/ load_sum

        delta = (pi2 - pi1) / average_load_curve.sum()
        confidence = 0.05
        self.volume_risk_var = self.calculate_var(delta, confidence)
        # self.volume_risk_chart = self.plot_histogram(delta, f'Volume risk var: {self.volume_risk_var:.2f} at $\\alpha = 0.05$\nVolume risk: {self.volume_risk}', '', '', self.volume_risk_var)
        self.volume_risk_chart = self.plot_histogram(delta,
                                                     f'Volume risk\nVaR($\\alpha={confidence*100:.0f}\\%$): {self.volume_risk_var:.2f}€',
                                                     'Volume risk', 'Frequency', self.volume_risk_var)

        return self

    # ----------------------------------- Plots ---------------------------------- #

    def generatePlots(self):
    
        # ------------------------------ Spot component ------------------------------ #

        # Seasonal curve and Log Price
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        self.spot_df.set_index('Date')[['LogPrice','Seasonal Curve']].plot(ax=ax)
        ax.set_title("Log Price with Seasonal fitted Curve")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.legend(["LogPrice", "Seasonal Curve"])
        self.figures.append(fig)

        # Seasonal Curve Projection
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        self.df_projection.set_index('Date')[['Seasonal Projection']].plot(ax=ax)
        ax.set_title("3 Years Projected Seasonal Curve")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        # ax.legend()
        self.figures.append(fig)

        # Stochastic component of log price
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        self.spot_df.set_index('Date')[['LogSeasonalResiduals']].plot(ax=ax)
        ax.set_title("Stochastic component of spot prices")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.legend(["LogSeasonalResiduals"]).set_visible(False)
        self.figures.append(fig)

        # Stochastic curve projection
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        self.st_projection.iloc[:365, :10].plot(ax=ax, legend=False)
        ax.set_title("1 Year Projected Stochastic Curve simulations")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        self.figures.append(fig)

        # 10 sample simpulations under P
        self.sample_simulations['Date'] = pd.date_range(
            start=self.spot_df['Date'].iloc[-1],
            periods=len(self.sample_simulations), freq='D')                 
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        self.sample_simulations.set_index('Date').iloc[:,:10].plot(ax=ax,legend=False)
        ax.set_xlabel(None)
        ax.set_title(r"Spot simulations under $\mathbb{P}$")
        ax.set_ylabel("Price")
        self.figures.append(fig)

        # Average price under P with forward
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        self.expected_price_p.plot(ax=ax)
        self.fw_df.set_index('Date')[['Close']].plot(ax=ax)
        ax.set_title(r'Expected Spot-Price under $\mathbb{P}$ vs. Forward Curve')
        ax.set_xlabel(None)
        ax.set_ylabel("Price")
        ax.legend(["Expected Spot-Price", "Observed Forward Curve"])
        self.figures.append(fig)
        #ax.legend(labels=['Observed Forward Curve','Theoretical Curve', 'Calibrated Curve'])

        # Stochastic component of log price
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        self.spot_df.set_index('Date')[['LogResiduals']].plot(ax=ax)
        ax.set_title("Stochastic component of spot prices")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.legend(["LogResiduals"]).set_visible(False)
        self.figures.append(fig)

        # Alphas
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        self.alphas.plot(ax=ax)
        ax.set_title(r'Calibrated $\alpha^*(t)$ function')
        ax.set_xlabel(None)
        #ax.set_ylabel("Price")
        #ax.legend().set_visible(False)
        self.figures.append(fig)

        # Calibrated forward curve 
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        self.fw_df.set_index('Date')[['Close','Theoretical curve','Calibrated curve']].plot(ax=ax)
        ax.set_title('Calibration of the forward curve')
        ax.set_xlabel(None)
        ax.set_ylabel("Price")
        ax.legend(labels=['Observed Forward Curve','Theoretical Curve', 'Calibrated Curve'])
        self.figures.append(fig)
        
        # Spot simulations under Q
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        self.simulatedPrice_all.iloc[:,:50].plot(legend=False, ax = ax)
        ax.set_title(r"Spot simulations under $\mathbb{Q}$")
        ax.set_xlabel(None)
        ax.set_ylabel("Price")
        self.figures.append(fig)

        # Average simulation path vs Forward curve
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        self.fw_df['Average simulations'] = self.simulatedPrice_all.mean(axis=1).values
        self.fw_df.set_index('Date')[['Close','Average simulations']].plot(ax=ax, label=None)
        ax.legend(labels=['Observed Forward Curve', f'Average path of {self.number_simulations} simulations'])
        ax.set_title('Average simulation path vs. Forward Curve\nafter calibration')
        ax.set_xlabel(None)
        ax.set_ylabel("Price")
        self.figures.append(fig)
        
        # ------------------------------ Load component ------------------------------ #
        
        # Print Load data
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        self.load_df_full.set_index('Date')[['Value']].plot(ax=ax, legend=False)
        #ax.legend(labels=['Forecasted Load Mwh', 'Seasonal fitted curve'])
        ax.set_title('Load consumptions')
        ax.set_xlabel(None)
        ax.set_ylabel("Load [Mwh]")
        self.figures.append(fig)

        # Forecasted Load and Seasional component fitted
        fig, ax = plt.subplots(figsize=(10, 5))
        self.simulatedLoad_all.set_index('Date')[['Value','Seasonal Curve']].plot(ax=ax)
        ax.legend(labels=['Load Mwh', 'Seasonal Curve Fitted'])
        ax.set_title('Load with fitted seasonal curve')
        ax.set_xlabel(None)
        ax.set_ylabel("Load [Mwh]")
        self.figures.append(fig)

        # Load residuals
        fig, ax = plt.subplots(figsize=(10, 5),constrained_layout=True)
        self.simulatedLoad_all.set_index('Date')[['Residuals']].plot(ax=ax, legend = False)
        ax.set_title('Load residuals')
        ax.set_xlabel(None)
        ax.set_ylabel("Load [Mwh]")
        self.figures.append(fig)

        # Residuals ACF and PACF
        lag = 100
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
        # ACF plot
        plot_acf(self.simulatedLoad_all['Residuals'], lags=lag, ax=axes[0], alpha=0.05, use_vlines=True, marker='')
        axes[0].set_title('ACF')
        # PACF plot
        plot_pacf(self.simulatedLoad_all['Residuals'], lags=lag, ax=axes[1], alpha=0.05, use_vlines=True, marker='')
        axes[1].set_title('PACF')
        # plt.tight_layout()
        self.figures.append(fig)

        # Residuals from AR(1)
        fig, ax = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)  # or use tight_layout later
        self.simulatedLoad_all.set_index('Date')[['Residuals2']].plot(ax=ax[0], legend=False)
        ax[0].set_title('Residuals from AR(1)')
        ax[0].set_xlabel(None)
        ax[0].set_ylabel("Load [MWh]")

        squared_residuals = self.simulatedLoad_all.set_index('Date')[['Residuals2']] ** 2
        squared_residuals.plot(ax=ax[1], legend=False)
        ax[1].set_title('Squared Residuals from AR(1)', pad=6)  # add a little padding
        ax[1].set_xlabel(None)
        ax[1].set_ylabel(None)
        self.figures.append(fig)

        # Monthly Variance fitted
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        self.volatility_df[['FittedVariance']].plot(style='-', ax = ax)
        self.volatility_df[['MonthlyVariance']].plot(style='o', ax=ax)
        ax.legend(labels=['Fitted variance', 'Observed avg. Monthly Variance'])
        ax.set_title('Fitting the monthly variance of the load')
        ax.set_xlabel("Monthly variance over a year")
        ax.set_ylabel("Load variance")
        self.figures.append(fig)

        # Plot simulated load
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        load = self.simulatedLoad
        load['Date'] = self.fw_df['Date'].iloc[:len(load)]
        load.set_index('Date').iloc[:,:100].plot(label='Simulated Load',ax=ax)
        ax.legend().remove()
        ax.set_title('Simulated load')
        ax.set_xlabel(None)
        ax.set_ylabel("Load [Mwh]")
        self.figures.append(fig)

        return self