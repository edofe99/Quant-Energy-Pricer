import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Hedging:
    
    def __init__(self, contract):
        
        self.contract = contract
        self.figures = []

    def calc_eta(self, contract_dates):    

        average_load_curve = self.contract.contractAverageLoad.reset_index(drop=True)
        eta = np.zeros(len(average_load_curve))

        # Make sure that the contract hedging ends on the last day of contract
        if contract_dates and contract_dates[-1] >= average_load_curve.index[-1]:
            contract_dates[-1] = average_load_curve.index[-1]+1  # Set last value to last index of df

        chart_title = ''
        for i in range(len(contract_dates) - 1):
            start, end = contract_dates[i], contract_dates[i+1]
            qty = average_load_curve.iloc[start:end].sum()/(end-start)
            eta[start:end] = average_load_curve.iloc[start:end].sum()/(end-start) 
            # chart_title += f'$\\eta_{{start}-{end}}={eta[start:end]}$'
            new_line = '\n' if (i+1) % 4 == 0 else ''
            chart_title += f'$\\eta_{{{start}-{end}}} = {qty:.2f}\\quad$' + new_line
            
        eta_df = pd.DataFrame()
        eta_df['Average Load'] = average_load_curve
        eta_df['eta'] = eta

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        eta_df.iloc[:, 0].plot(ax=ax1, label='Expected Load')
        eta_df.iloc[:, 1].plot(kind='area', ax=ax1, alpha=0.4, label=f'Hedging with {len(contract_dates)} forward contracts')
        ax1.set_xlabel('Contract days')
        ax1.set_title(chart_title)#, wrap=True)
        plt.tight_layout()  # Ensures title and chart fit well
        ax1.legend()

        # fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)
        # # Top chart: first column (line) + second column (area)
        # eta_df.iloc[:, 0].plot(ax=ax1, label='Expected Load')
        # eta_df.iloc[:, 1].plot(kind='area', ax=ax1, alpha=0.4, label=f'Hedging with {len(contract_dates)} forward contracts')
        # ax1.legend()

        # Bottom chart: delta between the two columns
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        delta_series = eta_df.iloc[:, 0] - eta_df.iloc[:, 1]
        delta_series.plot(ax=ax2, color='red')
        ax2.set_title('Hedging P/L')
        ax2.axhline(y=0, color='black', linestyle='--',alpha=0.4)
        ax2.set_xlabel('Contract days')
        # ax2.legend(None)
        
        self.figures.append(fig1)
        self.figures.append(fig2)

        return eta 

    def hedging(self, eta):

        load = self.contract.contractLoad.reset_index(drop=True)
        spot = self.contract.contractSpot.reset_index(drop=True)
        forward = self.contract.contractForward

        contractIndex = self.contract.contractIndex

        # --------------------------------- Iteration -------------------------------- #
        profit_array = np.zeros(len(contractIndex))
        use_forward = True

        ### ALTENRATIVE ITERATIVE METHOD
        # for i in range(len(contractIndex)):
        #     buy_price = forward[i] if use_forward else spot.iloc[i].mean()
        #     profit_array[i] = contract.optimalPrice*(contract.discount[i]*load.iloc[i].mean() ) - (
        #         contract.discount[i]*eta[i]*buy_price) + (
        #             contract.discount[i]* ((eta[i]-load.iloc[i,:].values)*spot.iloc[i,:].values).mean())

        # print(f'Expected profit: {profit_array.sum()}')

        # -------------------------------- Simulations ------------------------------- #
        profit_array = np.zeros((len(self.contract.contractIndex),self.contract.number_simulations)) # Arraw n.days x n.sims
        for i in range(len(self.contract.contractIndex)):
            buy_price = forward[i] if use_forward else spot.iloc[i,:].values
            profit_array[i,:] = (self.contract.optimalPrice * self.contract.discount[i] * load.iloc[i,:].values) - (
                self.contract.discount[i]*eta[i]*buy_price) + (
                    self.contract.discount[i]* ( (eta[i]-load.iloc[i,:].values) * spot.iloc[i,:].values)
                )

        # Convert to a DataFrame
        col_names = [f'Sim_Profit n.{i+1}' for i in range(self.contract.number_simulations)]
        profit_df = pd.DataFrame(profit_array, columns=col_names)

        nominal = True
        confidence = 0.95

        final_profit_scenarios = profit_df.sum(axis=0)/load.sum().values if nominal else profit_df.sum(axis=0)
        expected_profit = final_profit_scenarios.mean()

        var = self.contract.calculate_var(final_profit_scenarios,0.95)
        unit = f'{"€/MWh" if nominal else "€"}'

        # title_chart = f'Profit distribution using hedging\n Contract Price: {self.contract.optimalPrice:.2f}€\
        #         Final Contract Price: {self.contract.optimalPrice + self.contract.volume_risk_var:.2f}€\
        #         \nExpected profit: {expected_profit:.2f}{unit} | $\\alpha={1-confidence:.2f}$ $VAR(\\alpha)={var:.2f}{unit}$'
        
        title_chart = f'Profit distribution of a contract with price {self.contract.optimalPrice:.2f} using hedging\n' \
              f'Expected profit: {expected_profit:.2f}{unit} | ' \
              f'VAR(α={confidence*100:.0f}%): {var:.2f}{unit}'
            #   f'Contract Price: {self.optimalPrice:.2f}€ | ' \
            #   f'Final Contract Price: {self.optimalPrice+self.volume_risk_var:.2f}€\n' \



        fig = self.contract.plot_histogram(final_profit_scenarios,title_chart,f'Profit [{unit}]', 'Frequency',var)
        
        self.figures.append(fig)

        # profit_df.mean(axis=1).plot(title='Expected Profit evolution')
        
        # plt.show()
