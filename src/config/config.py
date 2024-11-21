class Config:
    #Dataset config
    KAGGLE_PATH='xhlulu/flickrfaceshq-dataset-nvidia-resized-256px'
    OUTPUT_DIR='output'

    #Input config
    INPUT_SIZE=(256, 256)
    INPUT_SHAPE=(256, 256, 3)
    MASK_SIZE=(92, 92)

    

    #Model config
    EPOCHS=100
    BATCH_SIZE=6
    GENERATOR_LEARNING_RATE=0.0002
    DISCRIMINATOR_LEARNING_RATE=0.0004
    BETA_1=0.5
