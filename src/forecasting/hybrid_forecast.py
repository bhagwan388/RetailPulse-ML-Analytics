import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error,mean_squared_error


PROPHET_FILE="outputs/predictions/prophet_forecast.csv"

LSTM_FILE="outputs/predictions/lstm_forecast.csv"


OUTPUT="outputs/predictions/hybrid_forecast.csv"

FIG="outputs/figures/hybrid_forecast.png"



print("Loading forecasts...")


prophet=pd.read_csv(PROPHET_FILE)

lstm=pd.read_csv(LSTM_FILE)



prophet["ds"]=pd.to_datetime(prophet["ds"])

lstm["ds"]=pd.to_datetime(lstm["ds"])



merged=pd.merge(
    prophet,
    lstm,
    on="ds",
    how="inner"
)



if len(merged)==0:

    print(
    "No historical overlap found."
    )

    print(
    "Using default weights."
    )

    future=pd.DataFrame()

    future["ds"]=lstm["ds"]

    future["prediction"]=(
        0.5*
        prophet["yhat"].values[:30]
        +
        0.5*
        lstm["prediction"].values[:30]
    )


else:

    print(
    "Optimizing weights..."
    )

    best_weight=0.5

    best_score=float("inf")


    for w in np.arange(0,1.1,0.1):

        pred=(
            w*merged["yhat"]
            +
            (1-w)*merged["prediction"]
        )

        score=np.mean(
            abs(
                pred-merged["prediction"]
            )
        )


        if score<best_score:

            best_score=score
            best_weight=w



    print(
    "Best Prophet weight:",
    best_weight
    )


    future=pd.DataFrame()

    future["ds"]=lstm["ds"]

    future["prediction"]=(
        best_weight*lstm["prediction"]
        +
        (1-best_weight)*lstm["prediction"]
    )



future.to_csv(
    OUTPUT,
    index=False
)



plt.figure(figsize=(12,5))

plt.plot(
    future["ds"],
    future["prediction"]
)


plt.title(
"Hybrid Prophet + LSTM Forecast"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(FIG)

plt.close()



print("\nHybrid forecast saved:")
print(OUTPUT)

print("Plot saved:")
print(FIG)

print(
"Hybrid forecasting completed successfully"
)