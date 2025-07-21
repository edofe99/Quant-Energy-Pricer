# ⚡️ Quant Energy Pricing

![Demo screenshot](/test/demo.png)

A tool for pricing full‑load energy contracts.

> A full‑load contract locks in a constant power delivery or energy volume over a set term. The buyer pays, and the seller delivers, the full amount regardless of actual usage.

## 🔧 Key Features

### 1. Optimal Contract Pricing  
- The required inputs are a csv file that contains a timeseries with spot prices folllowed by the observed Forward curve of a given day.
- On the input file name there should be a date in format "YYYY-MM-DD", representing the last day of spot-prices.
  - If this is not detected, the app will ask you to import a separate time-series with the observed Forward curve.
- In this repo I've included two files so that you can try the project:
  - `NatGas 2024-12-06.csv`: this is the file to select after clicking "Import Prices".
  - `Consumption.csv`: this is the file to select after clicking "Imoport Load". This file contains randomized values that reflects the consumption curve pattern observed in real-world data and literature.

### 2. Volume‐Risk Quantification  
- The volume risk is quantified in monetary value, so that you can have a measure on how much to charge on top of the risk-neutral price.
- Extrapolated trough stochastic simulation of clinent consumption.
### 3. Profit‐Distribution Analysis  
- The profit distribution is considered taking into account the calibrated simulations of spot prices and the simulations of load consumption.
- It is possible to observe the distribution also for different hedging strategies.

### 4. Hedging Strategy  
- You can try different strategies and see how increasing the amount of contracts actually decreases the risk.
- The input value requires to insert the time-range of contracts that are available to purchase (or liquid enough). So, assuming we can only buy two Forwards: one that will expire in 30 days and one that will expire in 90 days, then as input we would type `0,30,90`.

## 📌 Project Info

- **Documentation**: Detailed methodology and results in my MSc thesis (available September 2025, link will be inserted here).  
- **Sponsor**: Thanks to [Gruppo Dolomiti Energia](https://www.dolomitienergia.it/) for having me on this project.  
- **Special thanks**:  
  - [Fanone Enzo, PhD](https://www.linkedin.com/in/enzo-fanone-phd-3294592/) & [Traversi Alessandro](https://www.linkedin.com/in/alessandro-traversi-005ba51b/) at Dolomiti Energia Trading. 
  - Prof. [Gnoatto Alessandro](https://www.linkedin.com/in/alessandro-gnoatto-b184143b/) at [University of Verona](https://www.univr.it/en/university).

Thanks for stopping by 🙂
Feel welcome to open Issues, Pull Requests, or also to suggest new features.

---
This project was released under the MIT license.

