from data import DataSet
from model import *


if __name__ == "__main__":
    data = DataSet('exp')
    md = DuelingTripletNetwork()
    md = md.model
    md.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=[tf.keras.metrics.AUC(num_thresholds=3)])

    history = md.fit(
        x=data.train_input,
        y=data.train_output,
        verbose=1,
        epochs=100)
    
    result = md.evaluate(
       x=data.test_input,
       y=data.test_output
    )

    print(result)
