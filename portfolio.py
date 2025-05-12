import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

class PortfolioAnalysis:
    def __init__(self, portfolio, start_date="2000-01-01", end_date="2024-01-01", budget=1000, data_path="data/", freq="M"):
        self.portfolio = portfolio
        self.start_date = start_date
        self.end_date = end_date
        self.budget = budget
        self.data_path = data_path
        self.freq = freq
        self.n = len(portfolio)
        self.R = 0.006  # Risk-free return (FD/12)

        self.raw_data = None # pd DataFrame of closing prices
        self.data = None # numpy array of returns
        self.pd_data = None # resampled data pandas DataFrame

        # for plotting
        self.vals_to_plot = []
        self.dates_to_plot = []
        self.legend = []

        self._load_data()
        self._compute_returns()
    
    def _load_data(self):
        """Load stock data for the given portfolio and date range."""
        data = {}
        for stock in self.portfolio:
            filename = f"{self.data_path}{stock}{self.start_date}{self.end_date}.csv"

            if not os.path.exists(filename):
                s = yf.Ticker(stock)
                hist = s.history(start=self.start_date, end=self.end_date, interval="1d")
                # hist.index = hist.index.date
                hist.to_csv(filename)

            hist = pd.read_csv(filename, index_col=0, parse_dates=True)
            # hist.index = hist.index.date
            # hist.to_csv(filename)
            data[stock] = hist["Close"]

        self.raw_data = pd.DataFrame(data).dropna()
        self.raw_data.index = pd.to_datetime(self.raw_data.index)

    def _compute_returns(self):
        """Compute returns according to the frequency."""
        if self.freq == "M": # Monthly
            resampled = self.raw_data.resample('ME').last()
        elif self.freq == "BW": # Bi-weekly
            resampled = self.raw_data.resample('SMS').last()
        else: # Daily
            resampled = self.raw_data   
        self.raw_resampled = resampled.values  
        self.pd_data = resampled.pct_change().dropna()
        self.data = self.pd_data.values

    # AVERAGING/RETURN COVARIANCE ESTIMATION

    def EMA(self, span, smoothing):
        """Compute the Exponential Moving Average (EMA) of the portfolio."""
        # EMA formula = (alpha*d_today + (1-alpha)*EMA_yesterday)
        # where alpha = smoothing/1+days

        ema_returns = []
        ema_covs = []
        for i in range(len(self.data)):
            if i < span:
                ema_returns.append(None)
                ema_covs.append(None)
                continue

            if i == span:
                ema_returns.append(np.mean(self.data[:i], axis=0))
                ema_covs.append(np.cov(self.data[:i], rowvar=False))
                continue

            # calculate the EMA for returns
            alpha = smoothing / (1 + span)
            ema_returns.append((alpha * self.data[i]) + ((1 - alpha) * ema_returns[-1]))

            # calculate the EMA for covariance
            diff = self.data[i] - ema_returns[-1]
            cov = (1 - alpha) * ema_covs[-1] + alpha * np.outer(diff, diff)
            ema_covs.append(cov)
        return ema_returns, ema_covs
        
        # ema_returns = self.pd_data.ewm(span=span, adjust=False).mean()
        # # replace cov calc with the one here : https://in.mathworks.com/help/risk/estimate-var-using-parametric-methods.html
        # ema_covs = self.pd_data.ewm(span=span, adjust=False).cov(pairwise=True)

        # return_list = [row.to_numpy() for _, row in ema_returns.iterrows()]
        # cov_list = [ema_covs.loc[t].to_numpy() for t in self.pd_data.index]

        # return return_list, cov_list

    def SingleAverage(self):
        mu = np.mean(self.data, axis=0)
        cov = np.cov(self.data, rowvar=False)
        return mu, cov
    
    def RollingAverage(self, window):
        mu, cov = [], []
        for i in range(1, self.data.shape[0]+1):
            subset = self.data[max(0, i-window):i]
            if subset.shape[0] < 2:
                mu.append(None)
                cov.append(None)
                continue
            mu.append(np.mean(subset, axis=0))
            cov.append(np.cov(subset, rowvar=False))
        return mu, cov
    
    def CumRollingAverage(self):
        mu, cov = [], []
        for i in range(1, self.data.shape[0]+1):
            subset = self.data[:i]
            if subset.shape[0] < 2:
                mu.append(None)
                cov.append(None)
                continue
            mu.append(np.mean(subset, axis=0))
            cov.append(np.cov(subset, rowvar=False))
        return mu, cov

    def ARIMA(self): # PRETTY BAD. DOESN'T CONVERGE 99% OF THE TIME. FIX?
        """Compute the ARIMA model for the portfolio."""
        best_pqd = [None for _ in range(self.n)]
        best_aic = [np.inf for _ in range(self.n)]

        # find best pqd for each series
        for j in range(self.n):
            print(f"Fitting ARIMA for {self.portfolio[j]}...")
            # Check if the series is stationary
            result = adfuller(self.data[:, j])
            max_d = 0 if result[1] <= 0.05 else 2

            # Fit ARIMA model with different p, d, q values
            for p, d, q in itertools.product(range(3), range(max_d+1), range(3)):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(self.data[:, j], order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic[j]:
                        best_aic[j] = results.aic
                        best_pqd[j] = (p, d, q)
                except:
                    pass


        returns_list = []
        cov_list = []
        for i in range(1, self.data.shape[0]):
            if i < 10:
                returns_list.append(np.mean(self.data[:i], axis=0))
                cov_list.append(np.cov(self.data[:i], rowvar=False))
                continue

            mu = np.zeros(self.n)
            for j in range(self.n):
                print(f"Fitting ARIMA for {self.portfolio[j]} at time step {i}...")
                # Fit model for this time step
                p, d, q = best_pqd[j]
                try:
                    model = ARIMA(self.data[:i, j], order=(p, d, q))
                    results = model.fit()
                    mu[j] = results.forecast(1)[0]
                except:
                    mu[j] = np.mean(self.data[:i, j])

            returns_list.append(mu)
            cov_list.append(np.cov(self.data[max(0, i-30):i], rowvar=False))

        self.best_pqd = best_pqd
        return returns_list, cov_list

    def BlackScholes(self):
        # calculate mu_t and cov_t using averaging
        # use the model to predict price at t+1
        # use return t+1 = price_t+1 - price_t

        returns = []

        ema_returns, ema_covs = self.RollingAverage(10)

        for t in range(1, len(ema_returns)):
            if ema_returns[t-1] is None or ema_covs[t-1] is None or np.all(np.isnan(ema_covs[t-1])):
                returns.append(None)
                continue
            rt = []
            for i in range(len(self.portfolio)):
                mu_t = ema_returns[t-1][i]
                cov_t = ema_covs[t-1][i][i]
                s_t = self.raw_resampled[t][i]

                # calculate the BS estimate for t+1
                normal_sample = np.random.normal(0, 1)
                s_t1_est = s_t * np.exp((mu_t + 0.5 * (cov_t)**2) + (normal_sample * cov_t))


                # calculate the return
                return_t = (s_t1_est - s_t) / s_t

                rt.append(return_t)
            returns.append(np.array(rt))

        return returns, ema_covs

    def CAPM(self, indexfile): # PROBABLY HAS INDEXING ISSUES THAT ARE CURRENTLY IGNORED

        """Using CAPM to calculate the expected returns of the portfolio."""
        index = pd.read_csv(indexfile, index_col=0, parse_dates=True)
        index = index["Close"]

        # take intersection of index and portfolio data
        common_start = max(self.raw_data.index.min(), index.index.min())
        common_end = min(self.raw_data.index.max(), index.index.max())

        index = index[(index.index >= common_start) & (index.index <= common_end)]
        raw_data = self.raw_data[(self.raw_data.index >= common_start) & (self.raw_data.index <= common_end)]

        index = index.resample('ME').last()
        index = index.pct_change().dropna()
        raw_data = raw_data.resample('ME').last()
        raw_data = raw_data.pct_change().dropna()

        index, raw_data = index.values, raw_data.values
        
        # number of rows dropped from self.raw_data
        diff = self.data.shape[0] - raw_data.shape[0]

        # calculate the beta for each stock
        mu, cov = [], []
        window = 10
        for t in range(self.data.shape[0]):
            if t < diff:
                mu.append(None)
                cov.append(None)
                continue

            # calculate the beta for each stock
            index_slice = index[max(0, t-window):t]
            if index_slice.shape[0] < 2:
                mu.append(None)
                cov.append(None)
                continue

            beta = []
            for i in range(self.n):
                data_slice = raw_data[max(0, t-window):t, i]
                if data_slice.shape[0] < 2:
                    beta.append(None)
                    continue
                cov_matrix = np.cov(data_slice, index_slice)
                if cov_matrix[1][1] == 0:
                    beta.append(None)
                    continue
                beta.append(cov_matrix[0][1] / cov_matrix[1][1])

            # construct the mu vector
            mu_t = np.zeros(self.n)
            for i in range(self.n):
                if beta[i] is None:
                    mu_t = None
                    break
                Em = np.mean(index_slice)
                mu_t[i] = self.R + beta[i] * (Em - self.R)

            mu.append(mu_t)
            cov_t = np.cov(raw_data[max(0, t-window):t], rowvar=False)
            cov.append(cov_t)

        return mu, cov


    # PORTFOLIO CALCULATION

    def MVP(self, mu, cov):
        """Compute the Minimum Variance Portfolio (MVP) given the inverse covariance matrix."""
        invC = np.linalg.pinv(cov)
        dim = mu.shape[0]
        ones = np.ones(dim)
        wmin = (invC @ ones) / (ones.T @ invC @ ones)
        mumin = wmin.T @ mu
        return wmin, mumin

    def MVPforM(self, m, mu, cov):
        """Compute the MVP for a given target return m."""
        invC = np.linalg.pinv(cov)
        dim = mu.shape[0]
        ones = np.ones(dim)
        M = np.array([
            [mu.T @ invC @ mu, mu.T @ invC @ ones],
            [ones.T @ invC @ mu, ones.T @ invC @ ones]
        ])

        detM = np.linalg.det(M)
        if (detM == 0): # M not invertible so no solution exists
            return None, None, None

        a = (1/detM) * invC @ ((ones.T @ invC @ ones) * mu - (mu.T @ invC @ ones) * ones)
        b = (1/detM) * invC @ ((mu.T @ invC @ mu) * ones - (mu.T @ invC @ ones) * mu)

        w = (m*a) + b
        return w, a, b

    def MarketPfl(self, mu, cov):
        """Compute the Market Portfolio given the expected returns and covariance matrix."""
        invC = np.linalg.pinv(cov)
        dim = mu.shape[0]
        ones = np.ones(dim)
        wminR = (invC @ (mu - self.R*ones))/ (ones @ invC @ (mu - self.R*ones))
        return wminR

    # CALCULATIONS

    def get_values(self, weights):
        values = [self.budget]
        dates = [self.pd_data.index[0]]

        if len(weights) == 1: # single weight
            w = weights[0]
            for i in range(self.data.shape[0]):
                portfolio_return = w.dot(self.data[i])
                new_value = values[-1] * (1 + portfolio_return)
                if new_value < 0: new_value = 0
                values.append(new_value)
                dates.append(self.pd_data.index[i])
        
        else: # multiple weights
            for i in range(1, self.data.shape[0]):
                if weights[i-1] is None:
                    values.append(values[-1])
                    dates.append(self.pd_data.index[i])
                    continue
                portfolio_return = weights[i-1].dot(self.data[i])
                new_value = values[-1] * (1 + portfolio_return)
                if new_value < 0: new_value = 0
                values.append(new_value)
                dates.append(self.pd_data.index[i])

        return (values, dates)

    # PLOTTING

    def plot_returns(self):
        """Plot the returns of the portfolio."""

        num_cols = 3
        num_rows = len(self.portfolio)/3
        if len(self.portfolio) % 3 != 0:
            num_rows += 1

        fig, axs = plt.subplots(int(num_rows), num_cols, figsize=(15, 5*num_rows))
        fig.suptitle("Portfolio Returns", fontsize=16)
        axs = axs.flatten()
        for i, stock in enumerate(self.portfolio):
            ax = axs[i]
            ax.plot(self.pd_data.index, self.pd_data[stock], label=stock)
            ax.set_title(stock)
            ax.set_xlabel("Date")
            ax.set_ylabel("Returns")
            ax.legend()
            ax.grid()
        
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_prices(self):
        data = self.raw_data.values
        num_cols = 3
        num_rows = len(self.portfolio)/3
        if len(self.portfolio) % 3 != 0:
            num_rows += 1
        fig, axs = plt.subplots(int(num_rows), num_cols, figsize=(15, 5*num_rows))
        fig.suptitle("Portfolio Prices", fontsize=16)
        axs = axs.flatten()
        for i, stock in enumerate(self.portfolio):
            ax = axs[i]
            ax.plot(self.raw_data.index, data[:, i], label=stock)
            ax.set_title(stock)
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid()
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_stl(self, s):
        """Plot the STL decomposition of the portfolio returns."""
        for stock in self.portfolio:
            data = self.raw_data[stock]
            # print(data.shape)
            stl = STL(data, period=s)
            res = stl.fit()
            seasonal, trend, resid = res.seasonal, res.trend, res.resid
            
            plt.figure(figsize=(12, 12))
            plt.subplot(411)
            plt.plot(data, label='Original')
            plt.legend()
            plt.grid()
            plt.subplot(412)
            plt.plot(trend, label='Trend')
            plt.legend()
            plt.grid()
            plt.subplot(413)
            plt.plot(seasonal, label='Seasonal')
            plt.legend()
            plt.grid()
            plt.subplot(414)
            plt.plot(resid, label='Residual')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"images/STL_{stock}_{s}.png")
            plt.close()

    def plot_weights(self, weights):
        wlist = [[] for _ in range(self.n)]
        dates = []
        for i in range(len(weights)):
            if weights[i] is None:
                for j in range(self.n):
                    wlist[j].append(None)
                dates.append(self.pd_data.index[i])
                continue
            for j in range(len(weights[i])):
                wlist[j].append(weights[i][j])
            dates.append(self.pd_data.index[i])
        for i in range(self.n):
            plt.plot(dates, wlist[i], label=self.portfolio[i])
        plt.title("Portfolio Weights")
        plt.xlabel("Date")
        plt.ylabel("Weights")
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()

    def SML(self, wm, mu, cov, t):
        """Plot the Security Market Line (SML) given the market portfolio."""

        weights = np.random.dirichlet(np.ones(len(self.portfolio)), size=500)

        # calculating mu for w according to the RHS of (5.2) PTRM
        mu_w = np.dot(weights, mu)
        beta = (mu_w - self.R) / (wm.dot(mu) - self.R)
        calculated_mu = self.R + beta * (mu.dot(wm) - self.R)
        # cov_w = np.array([np.dot(w.T, cov).dot(wm) for w in weights])
        # var = wm.dot(cov).dot(wm)
        # beta = cov_w / var

        # getting actual mu values for t to t+1
        returns_t = self.data[t]
        mu_w = np.dot(weights, returns_t)

        plt.scatter(beta, mu_w, label='beta vs true mu')
        plt.title(f"Security Market Line")
        plt.scatter(beta, calculated_mu, label='beta vs calculated mu')
        plt.xlabel("Beta")
        plt.ylabel("Expected Return")
        plt.legend()
        plt.show()
        plt.close()

    def SML_difference(self, wm, w, mu):
        """Calculate the difference between the SML and the actual returns."""

        # calculating mu for w according to the RHS of (5.2) PTRM
        mu_w = np.dot(wm, mu)
        beta = (mu_w - self.R) / (wm.dot(mu) - self.R)
        calculated_mu = self.R + beta * (mu.dot(wm) - self.R)

        # getting actual mu values for t to t+1
        actual_mu = []
        for t in range(self.data.shape[0]):
            returns_t = self.data[t]
            actual_mu.append(np.dot(w, returns_t))
        
        difference = np.array(actual_mu) - calculated_mu
        dates = self.pd_data.index
        plt.plot(dates, difference)
        plt.title("Difference between SML and actual returns for MVP")
        plt.xlabel("Date")
        plt.ylabel("Difference")
        plt.grid()
        plt.show()
        plt.close()

    def BS_estimate(self, mu, cov, w): # black-scholes estimate
        """Estimate the Black-Scholes model for the portfolio."""

        if w is not None:
            mu_new, cov_new = [], []
            for i in range(len(mu)):
                if mu[i] is None or cov[i] is None or np.all(np.isnan(cov[i])):
                    mu_new.append(None)
                    cov_new.append(None)
                    continue
                mu_new.append(mu[i].dot(w))
                cov_new.append(cov[i].dot(w).dot(w.T))
            mu = mu_new
            cov = cov_new

        actual_s = []
        estimated_s = []

        # calculate the BS estimate for t+1 based on mu,cov estimates till t
        for t in range(1,len(mu)):
            s_t = self.raw_resampled[t].dot(w)
            s_t1 = self.raw_resampled[t+1].dot(w)

            mu_t = mu[t-1] # mu index 0 corresponds to t=1
            sig_t = cov[t-1]
            if mu_t is None or sig_t is None or np.all(np.isnan(sig_t)):
                continue

            # calculate the BS estimate for t+1
            normal_sample = np.random.normal(0, 1)
            s_t1_est = s_t * np.exp( (mu_t + 0.5 * (sig_t)**2) + (normal_sample * sig_t))

            actual_s.append(s_t1)
            estimated_s.append(s_t1_est)

            # print(f"t: {t}, mu: {mu_t}, sig: {sig_t}, actual: {s_t1}, estimated: {s_t1_est}")
        

        # plot the actual vs estimated
        plt.plot(range(len(actual_s)), actual_s, label="Actual")
        plt.plot(range(len(estimated_s)), estimated_s, label="Estimated")
        p = ""
        for i in range(len(self.portfolio)):
            p = p + self.portfolio[i] + " "
        plt.title("Black-Scholes Estimate vs Actual " + p)
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()

if __name__ == "__main__":
    portfolio, name = ["SBIN.NS", "HDFCBANK.NS"], "SBIN.NS HDFCBANK.NS"

    # portfolio = ["INFY", "CIPLA.NS", "LNT", "TCS.NS", "HDFCBANK.NS", "SBIN.NS"]
    # portfolio = ["^NSEI"]
    analysis = PortfolioAnalysis(portfolio)

    # analysis.plot_prices()
    # analysis.plot_stl(252)

    # print("Values:", values[0], "\n\n", values[-1])
    # plt.plot(dates, values)
    # plt.title("Portfolio Value Over Time")
    # plt.xlabel("Date")
    # plt.ylabel("Portfolio Value")
    # plt.grid()
    # plt.show()

    returns, covs = analysis.CumRollingAverage()
    # print(returns[50], covs[50])
    w = analysis.MarketPfl(returns[50], covs[50])
    analysis.SML(w, returns[50], covs[50], 50)

    # analysis.plot_returns()

    # mu, cov = analysis.SingleAverage()
    # wMPL = analysis.MarketPfl(mu, cov)
    # wm, _ = analysis.MVP(mu, cov)
    # analysis.SML_difference(wMPL, wm, mu)

    # mu, cov = analysis.RollingAverage(10)
    # print(type(mu), type(cov))
    # print(len(mu))
    # print(len(analysis.raw_resampled))
    # mvp, _ = analysis.MVP(mu[1], cov[1])
    # analysis.BS_estimate(mu, cov, mvp)
