from tensorflow.keras.optimizers import Adam
from src.model import build_model
# Assuming data generators from data_prep

model = build_model(num_classes=3)
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, epochs=20, validation_data=val_gen)
model.save('models/trained_model.h5')

# Plot accuracy/loss
plt.plot(history.history['accuracy']); plt.savefig('results/plots/acc_plot.png')