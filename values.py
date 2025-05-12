from portfolio import PortfolioAnalysis
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    averaging = ["EMA", "RollingAverage", "SingleAverage", "CumRollingAverage"]

    # portfolio, name = ["CIPLA.NS", "LNT"], "CIPLA.NS LNT"
    # portfolio, name = ["TCS.NS", "INFY"], "TCS.NS INFY"
    # portfolio, name = ["SBIN.NS", "HDFCBANK.NS"], "SBIN.NS HDFCBANK.NS"
    portfolio, name = ["INFY", "CIPLA.NS", "LNT", "TCS.NS", "HDFCBANK.NS", "SBIN.NS"], "All"

    analysis = PortfolioAnalysis(portfolio)
    # print(analysis.data.shape)

    muSA, covSA = analysis.SingleAverage()
    muCum, covCum = analysis.CumRollingAverage()
    muRA, covRA = analysis.RollingAverage(10)
    muEMA, covEMA = analysis.EMA(10, 2)
    muCAPM, covCAPM = analysis.CAPM("data/^NSEI2000-01-012024-01-01.csv")
    # print("Starting ARIMA")
    muARIMA, covARIMA = analysis.ARIMA()
    # print("Starting Black-Scholes")
    # muBS, covBS = analysis.BlackScholes()

    analysis.legend = ["Single Average", "Cumulative", "Rolling(10)", "Weighted", "CAPM", "ARIMA"] #, "Black-Scholes"]
    MVP_plots, M_plots, MPFL_plots, R2_plots = [], [], [], []

    w, m = analysis.MVP(muSA, covSA)
    MVP_plots.append(analysis.get_values([w]))    
    w, _, _ = analysis.MVPforM(2*m, muSA, covSA)
    M_plots.append(analysis.get_values([w]))
    w = analysis.MarketPfl(muSA, covSA)
    MPFL_plots.append(analysis.get_values([w]))

    # 2r stuff
    muMPL = muSA.dot(w)
    lambdaMPL = (2*analysis.R - analysis.R) / (muMPL - analysis.R)
    w2r = np.array([ (1 - lambdaMPL) + (lambdaMPL * wi) for wi in w])
    R2_plots.append(analysis.get_values([w2r]))

    for rets, covs in [(muCum, covCum), (muRA, covRA), (muEMA, covEMA), (muCAPM, covCAPM), (muARIMA, covARIMA)]: #, (muBS, covBS)]:
        # for each time step:
        weightsMVP, weightsM, weightsMPFL, weights2r = [], [], [], []
        for i in range(len(rets)):
            if rets[i] is None or covs[i] is None or np.all(np.isnan(covs[i])):
                weightsMVP.append(None)
                weightsM.append(None)
                weightsMPFL.append(None)
                weights2r.append(None)
                continue
            w, m = analysis.MVP(rets[i], covs[i])
            weightsMVP.append(w)
            w, _, _ = analysis.MVPforM(2*m, rets[i], covs[i])
            weightsM.append(w)
            w = analysis.MarketPfl(rets[i], covs[i])
            weightsMPFL.append(w)

            # 2r stuff
            muMPL = rets[i].dot(w)
            lambdaMPL = (2*analysis.R - analysis.R) / (muMPL - analysis.R)
            w2r = np.array([ (1 - lambdaMPL) + (lambdaMPL * wi) for wi in w])
            weights2r.append(w2r)
        MVP_plots.append(analysis.get_values(weightsMVP))
        M_plots.append(analysis.get_values(weightsM))
        MPFL_plots.append(analysis.get_values(weightsMPFL))
        R2_plots.append(analysis.get_values(weights2r))

        # analysis.plot_weights(weightsMVP)
        # analysis.plot_weights(weightsM)


    # Plotting
    for i in range(len(MVP_plots)):
        values, dates = MVP_plots[i]
        plt.plot(dates, values, label=analysis.legend[i])
    plt.title(name + " Minimum Variance Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()

    for i in range(len(M_plots)):
        values, dates = M_plots[i]
        plt.plot(dates, values, label=analysis.legend[i])
    plt.title(name + " Portfolio for M")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()

    for i in range(len(MPFL_plots)):
        values, dates = MPFL_plots[i]
        plt.plot(dates, values, label=analysis.legend[i])
    plt.title(name + " Market Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.show()

    for i in range(len(R2_plots)):
        values, dates = R2_plots[i]
        plt.plot(dates, values, label=analysis.legend[i])
    plt.title(name + " Portfolio for 2r")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.show()
