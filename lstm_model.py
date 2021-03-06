from keras.layers import Dense,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model ,save_model
from keras.optimizers import Adam




class lstm_model():
    def __init__(self,nb_categories,sampling_rate,feature_dim):
        self.nb_categories=nb_categories
        self.sequence_shape=(sampling_rate,feature_dim)
        self.model= Sequential()
        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.add(LSTM(2048,return_sequences=False,input_shape=self.sequence_shape,dropout=0.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_categories, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        print(self.model.summary())
    def getmodel(self):
        return self.model


if __name__ == "__main__":
    model = lstm_model(5,40,2048)
