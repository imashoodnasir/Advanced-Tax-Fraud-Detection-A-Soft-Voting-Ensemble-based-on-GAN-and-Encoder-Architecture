import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import tensorflow as tf

# Step 1: Load Dataset
def load_dataset():
    # Replace with actual dataset loading
    X = np.random.rand(1000, 20)  # 1000 samples, 20 features
    y = np.random.randint(0, 2, 1000)  # Binary target
    return X, y

# Step 2: Split dataset into training, validation, and test sets
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Step 3: Define CGAN model
def define_cgan(input_dim):
    # Generator
    generator_input = Input(shape=(input_dim,))
    generator_output = Dense(input_dim, activation='relu')(generator_input)
    generator = Model(generator_input, generator_output)

    # Discriminator
    discriminator_input = Input(shape=(input_dim,))
    discriminator_output = Dense(1, activation='sigmoid')(discriminator_input)
    discriminator = Model(discriminator_input, discriminator_output)
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

    # CGAN
    discriminator.trainable = False
    cgan_input = Input(shape=(input_dim,))
    cgan_output = discriminator(generator(cgan_input))
    cgan = Model(cgan_input, cgan_output)
    cgan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

    return generator, discriminator, cgan

# Step 4: Train CGAN
def train_cgan(generator, discriminator, cgan, X_train, epochs=1000, batch_size=32):
    for epoch in range(epochs):
        # Select random real samples
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_samples = X_train[idx]

        # Generate fake samples
        noise = np.random.normal(0, 1, (batch_size, X_train.shape[1]))
        fake_samples = generator.predict(noise)

        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_samples, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_samples, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator (via CGAN)
        g_loss = cgan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss[0]} | G Loss: {g_loss}")

# Step 5: Generate synthetic data
def generate_synthetic_data(generator, num_samples, input_dim):
    noise = np.random.normal(0, 1, (num_samples, input_dim))
    return generator.predict(noise)

# Step 6: Train classifier and evaluate
def train_and_evaluate_classifier(X_train, X_val, X_test, y_train, y_val, y_test):
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    # Validation
    y_val_pred = classifier.predict(X_val)
    print("Validation Metrics:")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred)}")
    print(f"F1-Score: {f1_score(y_val, y_val_pred)}")

    # Test
    y_test_pred = classifier.predict(X_test)
    print("Test Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
    print(f"F1-Score: {f1_score(y_test, y_test_pred)}")
    print(f"Precision: {precision_score(y_test, y_test_pred)}")
    print(f"Recall: {recall_score(y_test, y_test_pred)}")

# Main pipeline
def main():
    # Load and preprocess data
    X, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Define and train CGAN
    generator, discriminator, cgan = define_cgan(X_train.shape[1])
    train_cgan(generator, discriminator, cgan, X_train)

    # Generate synthetic data
    synthetic_data = generate_synthetic_data(generator, num_samples=500, input_dim=X_train.shape[1])

    # Combine real and synthetic data
    X_combined = np.vstack((X_train, synthetic_data))
    y_combined = np.hstack((y_train, np.random.randint(0, 2, synthetic_data.shape[0])))

    # Train and evaluate classifier
    train_and_evaluate_classifier(X_combined, X_val, X_test, y_combined, y_val, y_test)

if __name__ == "__main__":
    main()
