import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


DATA_PATH="data/processed/daily_sales_clean.csv"

MODEL_PATH="models/forecasting/lstm_model.pth"

OUTPUT_PATH="outputs/predictions/lstm_forecast.csv"

FIG_PATH="outputs/figures/lstm_forecast.png"


device="cpu"


df=pd.read_csv(DATA_PATH)

df["ds"]=pd.to_datetime(df["ds"])

df=df.sort_values("ds")


print("Dataset size:",len(df))


values=df["y"].values.reshape(-1,1)


scaler=MinMaxScaler()

scaled=scaler.fit_transform(values)


window=30


X=[]
y=[]


for i in range(window,len(scaled)):

    X.append(
        scaled[i-window:i]
    )

    y.append(
        scaled[i]
    )


X=torch.tensor(
    np.array(X),
    dtype=torch.float32
)

y=torch.tensor(
    np.array(y),
    dtype=torch.float32
)



class LSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm=nn.LSTM(
            1,
            64,
            batch_first=True
        )

        self.fc=nn.Linear(
            64,
            1
        )


    def forward(self,x):

        out,_=self.lstm(x)

        return self.fc(out[:,-1,:])



model=LSTM().to(device)


optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


loss_fn=nn.MSELoss()



for epoch in range(100):

    model.train()

    optimizer.zero_grad()

    pred=model(X)

    loss=loss_fn(
        pred,
        y
    )

    loss.backward()

    optimizer.step()


    if (epoch+1)%10==0:

        print(
            f"Epoch {epoch+1}/100 Loss:{loss.item():.6f}"
        )



torch.save(
    model.state_dict(),
    MODEL_PATH
)


print("Model saved")



# future prediction

model.eval()


sequence=scaled[-window:]

future=[]


for i in range(30):

    inp=torch.tensor(
        sequence.reshape(1,window,1),
        dtype=torch.float32
    )


    with torch.no_grad():

        pred=model(inp).numpy()


    future.append(pred[0][0])


    sequence=np.append(
        sequence[1:],
        pred
    )



future=scaler.inverse_transform(
    np.array(future).reshape(-1,1)
)


dates=pd.date_range(
    df["ds"].max()+pd.Timedelta(days=1),
    periods=30
)



output=pd.DataFrame(
    {
        "ds":dates,
        "prediction":future.flatten()
    }
)


output.to_csv(
    OUTPUT_PATH,
    index=False
)


plt.figure(figsize=(12,5))

plt.plot(
    output["ds"],
    output["prediction"]
)

plt.title(
"LSTM 30 Day Forecast"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(FIG_PATH)

plt.close()


print("LSTM forecasting completed successfully")