# 📊 Solana Price Prediction Using LSTM: An In-Depth Analysis

## 🌟 Overview

The provided code implements a **time series forecasting pipeline** for predicting the future prices of Solana (SOL) cryptocurrency. It uses:

- **Pandas & Numpy** for data manipulation.

- **PyTorch** for deep learning.

- **Matplotlib** for visualization.

- **Sklearn** for data scaling.

The model is built around a **Stacked LSTM (Long Short-Term Memory)** network and is trained on historical Solana price data to predict **future prices.**

# 🔧 1. Environment & Deterministic Setup

```
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## ✔️ Why?

This ensures **reproducibility.** Machine learning can be **non-deterministic**, especially on GPUs. Setting the same seed allows the same results to be obtained across multiple runs.

# 🖥️ 2. Device Selection

`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

## Why?

Trains the model on **GPU if available**, significantly speeding up training for large datasets.

# 3. Data Preparation

## 📈 Key Steps:

- Date Parsing & Sorting:

```
solana_data['Date'] = pd.to_datetime(solana_data['Date'], format='%m/%d/%Y')
solana_data = solana_data.iloc[::-1]
```

✅ Ensures the data is **chronologically ordered** from past to present.

- **Feature Engineering:**

  - **Momentum_7d:** 7-day percentage change.
  - **Momentum_14d:** 14-day percentage change.

- **Data Cleaning:**

```
if data.isnull().any().any():
    data = data.fillna(method='ffill').fillna(method='bfill')
```

✅ This step **fills missing values** using forward and backward filling.

- **Scaling:**

```
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
```

✅ **MinMaxScaler** scales features between 0 and 1, crucial for stable neural network training.

- **Sequence Creation:**
  - For each sample:
    - **X:**Sequence of past 180 days.
    - **y:**The next day's feature vector.

# 🔥 **4. LSTM Model Architecture**

## 🧱 **Architecture:**

| Layer                       | Details                                     |
| --------------------------- | ------------------------------------------- |
| LSTM Layer 1                | 150 hidden units, input size = num_features |
| LSTM Layer 2                | 150 hidden units                            |
| Dropout                     | 0.3 dropout rate                            |
| Fully Connected Layer (FC1) | Linear(150 → 100), ReLU activation          |
| Fully Connected Layer (FC2) | Linear(100 → num_features)                  |

## ✔️ **Key Points:**

- **Stacked LSTM:**
  Two LSTM layers allow the model to **capture deeper temporal dependencies.**
- **Dropout:** Reduces overfitting by randomly dropping neurons during training.
- **Fully Connected Layers:** Map the learned hidden representations to the final **output vector** (all features).
- **Output:** Predicts **all features** (not just price), enabling multi-variate predictions.

# 🏋️ 5. Training Loop

## 🌀 Core Concepts:

- **Loss:** **MSELoss**, common for regression tasks.
- **Optimizer:** **Adam**, robust and adaptive learning rate optimizer.
- **Scheduler:**

`scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)`

✅ Automatically reduces the learning rate if loss plateaus, helping fine-tune learning.

- **Gradient Clipping:**

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

✅ Prevents **exploding gradients**, especially important in RNNs like LSTM.

- **Early Stopping**:

If the validation loss doesn't improve for 5 consecutive epochs, training stops early.

## 💾 Model Checkpoint:

`torch.save(model.state_dict(), 'best_model.pth')`

✅ Saves the **best model** during training to avoid overfitting at later epochs.

# 🔮 6. Forecasting Future Prices

## 🔍 Key Steps:

1. **Start with the last known sequence (180 days).**
2. **Iteratively predict the next day:**

- Feed the sequence to the model.
- Add **Gaussian noise (mean=0, std=0.003)** to **simulate real-world randomness.**
- Update the sequence by sliding the window forward.

3. **Repeat for 180 days.**

## 🛡️ Noise Injection:

`noise = np.random.normal(0, 0.003, size=num_features)`

✅ Adds slight randomness to avoid **overconfidence** and **simulate market noise.**

## 📅 Example Output:

`print(f"Predicted price for March 30, 2025: ${first_predicted_price:.4f}")`

Provides **immediate feedback** on the first forecasted price.

# 🖼️ 7. Plotting

- The plot:
  - **Blue Line:** Historical prices.
  - **Red Dashed Line:** Forecasted prices.

This **visualizes the transition** from historical data to future predictions, offering intuitive insight into model performance.

# 📈 8. Output & Export

The model **saves predictions** to a CSV file:

`files/solana_predictions.csv`

- Columns:
  - Date
  - Predicted Price

This is useful for **post-analysis**, **external plotting**, or **deployment**.

# 🚀 Potential Improvements

## 🛠️ Model Enhancements:

- **Bidirectional LSTM:** Capture **past & future context.**
- **Attention Mechanisms:** Focus on important timesteps.
- **Transformer Models:** Explore modern architectures like **Time Series Transformers.**

## 📅 Data Handling:

- **More Features:** Include macroeconomic indicators or social media sentiment.
- **Outlier Handling:** Apply robust statistics to handle sudden crypto spikes.

## 🧪 Training Strategy:

- **Cross-Validation:** Ensure the model generalizes well.
- **Hyperparameter Search:** Use grid/random search for optimal tuning.

# 🧹 Code Optimizations:

- **Modularization:** Break into smaller modules for clarity.
- **Logging:** Integrate **TensorBoard** or **Weights & Biases** for real-time monitoring.

## 📌 Conclusion

This pipeline is a **robust implementation** of time series forecasting using deep learning. It is:

- Reproducible
- Modular
- Scalable

It's a solid base for **further experimentation** and **refinement** as you seek to predict Solana's crypto actions

## However...

As I've stated on the first part of the project. Crypto doesn't really work like that. What this model does is analyse patterns and make a prediction on what's going to happen **next.**

But crypto relies on a lot of different aspects like;

- News
- Politics
- Worldwide events
- Supply & Demand
- Competing cryptocurrencies
- And even **investor's sentiment!**

So that if we lived in a world where crypto relied on only patterns, then this model would work. It concludes this project's requirements, however it's useless in a real-life scenario...

# 📂 Files :]

- final-model.ipynb is the **final** model.
- old_model_attempts.ipynb is the slight bit of tomfoolery I've done before getting to the final version.
- files is the folder that the model needs to function. It includes **spreadsheets** and **a python code** I used to refine some of the spreadsheets.
