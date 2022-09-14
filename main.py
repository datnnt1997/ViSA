from visa import TRAIN, TEST, LOGGER, TRAIN_WITH_RAY

import sys

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        LOGGER.info("Start TRAIN process...")
        TRAIN()
    if sys.argv[1] == 'hparam_tuning':
        LOGGER.info("Start Hyperparameter tuning process...")
        TRAIN_WITH_RAY()
    elif sys.argv[1] == 'test':
        LOGGER.info("Start TEST process...")
        TEST()
    elif sys.argv[1] == 'predict':
        LOGGER.info("Start PREDICT process...")
        raise NotImplementedError
    elif sys.argv[1] == 'demo':
        LOGGER.info("Start PREDICT process...")
        raise NotImplementedError
else:
        LOGGER.error(f'[ERROR] - `{sys.argv[1]}` not found!!!')