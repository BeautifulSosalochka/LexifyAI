import numpy as np
import pandas as pd
import os
from glob import glob
import json
import tensorflow as tf
from keras import layers, models
import neptune as neptune
from keras import EarlyStopping
from keras import optimizers
from keras import losses
from keras import regularizers


class AlmostPurityQNetwork:
    def __init__(self, input_shape, n_actions, learning_rate=0.0001, l2_lambda=0.01):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.model = self.build_model()
        self.compile_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='linear', input_shape=self.input_shape))
        model.add(layers.LeakyReLU(alpha=0.1))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.1))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.1))
        attention = layers.Conv2D(64, (3, 3), strides=2, padding='same')(model.output)
        attention = layers.LeakyReLU(alpha=0.1)(attention)
        attention = layers.Flatten()(attention)
        x = layers.Flatten()(model.output)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='linear')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(128, activation='linear')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(64, activation='linear')(x)
        x = layers.LeakyReLU()(x)

        # concat = Concatenate()([flattened_state, action_input])

        combined = layers.Concatenate()([x, attention])
        combined = layers.Dense(128, activation='linear', kernel_regularizer=regularizers.l2(self.l2_lambda))(combined)
        combined = layers.LeakyReLU(alpha=0.1)(combined)

        output = layers.Dense(self.n_actions, activation='softmax',
                              kernel_regularizer=regularizers.l2(self.l2_lambda))(combined)

        return models.Model(inputs=model.input, outputs=output)

    def compile_model(self):
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        loss = losses.CategoricalCrossentropy()

        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def __call__(self, inputs):
        return self.model.predict(inputs)

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model = models.load_model(file_path)

    def __str__(self):
        return self.model.summary()


class DataLoader:
    def __init__(self, train_path, val_path, test_path):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        train_data = self.load_json_data(self.train_path)
        val_data = self.load_json_data(self.val_path)
        test_data = self.load_json_data(self.test_path)

    @staticmethod
    def load_json_data(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data)

    def get_data(self):
        return self.train_data, self.val_data, self.test_data

    def __getitem__(self, idx):
        if idx < len(self.train_data):
            return self.train_data.iloc[idx]
        elif idx < len(self.train_data) + len(self.val_data):
            return self.val_data.iloc[idx - len(self.train_data)]
        elif idx < len(self.train_data) + len(self.val_data) + len(self.test_data):
            return self.test_data.iloc[idx - len(self.train_data) - len(self.val_data)]
        else:
            raise IndexError("Error: index is going outside...")

    def __len__(self):
        return len(self.train_data) + len(self.val_data) + len(self.test_data)

    def __str__(self):
        return (f"DataLoader: {len(self.train_data)} train samples, "
                f"{len(self.val_data)} val samples, {len(self.test_data)} test samples")

# class AutoEncoder(nn.Module):
#     def __init__(self, input_size, output_size, latent_size):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, latent_size),
#             nn.Sigmoid()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_size, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, output_size),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         latent = self.encoder(x)
#         return self.decoder(latent)


# class LSTM(nn.Module):
#     def __init__(self, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE):
#         super().__init__()
#         self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
#         self.linear = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
#
#     def forward(self, input_data):
#         lstm_out, _ = self.lstm(input_data)
#         predictions = self.linear(lstm_out)
#         return predictions

# class Training:
#     def __init__(self, model, train_loader, criterion, optimizer):
#         self.model = model
#         self.train_loader = train_loader
#         self.criterion = criterion
#         self.optimizer = optimizer
#
#     def _train_one(self, model, data, criterion, optimizer):
#         model.train()
#         input_data, target = data
#         input_data, target = input_data.to(device).float(), target.to(device).float()
#
#         output = model(input_data)
#         loss = criterion(output, target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         return loss.item()
#
#     def _train_loop(self, model, train_loader, criterion, optimizer):
#         model.train()
#         history = {'train_loss': []}
#         loss = 1
#         epoch = 0
#
#         while True:
#             epoch += 1
#             train_loss = 0
#             for data in train_loader:
#                 ls = self._train_one(model, data, criterion, optimizer)
#                 train_loss += ls
#             train_loss /= len(train_loader)
#             history['train_loss'].append(train_loss)
#
#             print(f'\rEpoch : {epoch}, Loss: {train_loss:.5f}, Lowest Loss: {loss:.5f}', end='')
#
#             if train_loss < loss:
#                 loss = train_loss
#                 torch.save(model.state_dict(), 'model.pth')
#             if train_loss < 0.01:
#                 break
#
#             # Waste too much time and loss is ok
#             if epoch > 100 and train_loss < 0.01:
#                 break
#             # F
#             if epoch > 200:
#                 break
#
#         return history
#
#     def train(self):
#         history = self._train_loop(self.model, self.train_loader, self.criterion, self.optimizer)
#         self._plot_loss(history)

class Trainer:
    def __init__(self, model, train_data, val_data, config):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.run = None
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    def initialize_neptune(self, project_name, api_token):
        self.run = neptune.init(project=project_name, api_token=api_token)

    def train(self, epochs=100):
        history = self.model.fit(
            self.train_data['X'], self.train_data['y'],
            epochs=epochs,
            validation_data=(self.val_data['X'], self.val_data['y']),
            callbacks=[self.early_stopping]
        )
        self.log_metrics(history)

    def log_metrics(self, history):
        for epoch, logs in enumerate(history.history['loss']):
            self.run['train/loss'].log(logs)
            self.run['train/val_loss'].log(history.history['val_loss'][epoch])

    def save_model(self, file_path):
        self.model.save(file_path)

    def __call__(self, data):
        return self.model(data)

    def __str__(self):
        return f"Trainer with model: {self.model} and Neptune logging."

# alpha = 0.6
# beta = 0.4
# prioritized_replay_buffer = []
#
# # Early Stopping callback
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
#
# class EarlyStopping:
#     def __init__(self, patience=10, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.best_reward = -np.inf
#         self.wait = 0
#
#     def step(self, current_reward):
#         if current_reward - self.best_reward > self.min_delta:
#             self.best_reward = current_reward
#             self.wait = 0
#         else:
#             self.wait += 1
#
#         if self.wait >= self.patience:
#             return True
#         return False

class NeptuneConfig:
    def __init__(self, project_name, api_token, model_name, tags=None):
        self.project_name = project_name
        self.api_token = api_token
        self.model_name = model_name
        self.run = None
        self.tags = tags or []

    def init_run(self, experiment_name=None, custom_params=None):
        self.run = neptune.init(
            project=self.project_name,
            api_token=self.api_token,
            name=experiment_name or self.model_name,
            tags=self.tags,
            source_files=["*.py"]  # Автоматически логирует текущие скрипты
        )
        if custom_params:
            self.run["parameters"] = custom_params

    def log_hyperparameters(self, hyperparams: dict):
        for param, value in hyperparams.items():
            self.run[f"hyperparameters/{param}"] = value

    def log_metrics(self, metrics: dict, step=None):
        for metric, value in metrics.items():
            self.run[f"metrics/{metric}"].log(value, step=step)

    def log_model(self, model, file_path='model.h5'):
        model.save(file_path)
        self.run["artifacts/model"].upload(file_path)

    def log_custom_data(self, data_name: str, data):
        self.run[f"custom_data/{data_name}"].upload(data)

    def log_artifacts(self, directory_path: str):
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                full_path = os.path.join(root, file)
                self.run[f"artifacts/{file}"].upload(full_path)

    def stop(self):
        if self.run:
            self.run.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def log_experiment_info(self, description, author):
        self.run["sys/description"] = description
        self.run["sys/author"] = author

    def monitor_system(self, interval=10):
        from neptune.integrations.tensorflow_keras import NeptuneCallback
        from keras import callbacks

        return [
            NeptuneCallback(run=self.run, log_on_batch=True),
            callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.run['sys/resources'].log(
                {"epoch": epoch, "logs": logs}))
        ]



