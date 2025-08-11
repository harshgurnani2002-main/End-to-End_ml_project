import logging
import os
from datetime import datetime
from exception import CustomException
import sys

# Timestamped log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Logs directory path
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)  # Create logs directory if not exists

# Full log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure basic logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


if __name__=='__main__':
    try :
        a=1/0
    except Exception as e:
        logging.info('divide by zero error')
        raise CustomException(e,sys)