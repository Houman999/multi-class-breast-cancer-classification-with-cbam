# Libeary
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model

def build_model_with_cbam(input_shape=(224, 224, 3), num_classes=8):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = CBAM()(x)
    x = Flatten()(x)
    
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)


    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


# Train Model 
model_with_cbam = build_model_with_cbam()
model_with_cbam.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

history = model_with_cbam.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[early_stopping, lr_scheduler]
)